#define _GNU_SOURCE
#define _FILE_OFFSET_BITS 64

#include <errno.h>
#include <fcntl.h>
#include <ftw.h>
#include <limits.h>
#include <linux/fiemap.h>
#include <linux/fs.h>
#include <linux/io_uring.h>
#include <linux/stat.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/sysmacros.h>
#include <sys/uio.h>
#include <sys/utsname.h>
#include <time.h>
#include <unistd.h>

#ifndef RWF_DONTCACHE
#define RWF_DONTCACHE 0x00000080
#endif

#define DEFAULT_FILE_LIMIT 10000
#define REPORT_PATH "benchmark_report.txt"
#define IO_URING_ENTRIES 64

struct FileEntry {
    char *path;
    size_t size;
    uint64_t physical;
    size_t memory_offset;
    size_t scan_index;
};

struct BenchmarkResult {
    const char *name;
    bool supported;
    size_t files_read;
    size_t bytes_read;
    double seconds;
    double mib_per_second;
    double user_seconds;
    double system_seconds;
    long minor_faults;
    long major_faults;
    bool device_stats_available;
    unsigned long long device_read_bytes;
    unsigned long long device_read_ticks;
    unsigned long long device_io_ticks;
    char note[256];
};

struct DeviceStats {
    bool available;
    unsigned long long read_ios;
    unsigned long long read_sectors;
    unsigned long long read_ticks;
    unsigned long long io_ticks;
};

struct IoUring {
    int fd;
    unsigned sq_entries_count;
    unsigned *sq_head;
    unsigned *sq_tail;
    unsigned *sq_ring_mask;
    unsigned *sq_ring_entries;
    unsigned *sq_flags;
    unsigned *sq_dropped;
    unsigned *sq_array;
    struct io_uring_sqe *sqes;
    unsigned *cq_head;
    unsigned *cq_tail;
    unsigned *cq_ring_mask;
    unsigned *cq_ring_entries;
    unsigned *cq_overflow;
    struct io_uring_cqe *cqes;
    void *sq_ring_ptr;
    void *cq_ring_ptr;
    size_t sq_ring_size;
    size_t cq_ring_size;
};

struct IoJob {
    int fd;
    size_t file_index;
    size_t completed;
};

struct DirectArena {
    char *memory;
    size_t *offsets;
    size_t alignment;
    size_t size;
    bool alignment_from_statx;
};

static struct FileEntry *files;
static size_t file_count;
static size_t file_capacity;
static size_t file_limit = DEFAULT_FILE_LIMIT;
static size_t total_size;
static dev_t source_device;
static bool collection_limit_reached;
static double scan_seconds;

static double elapsed_seconds(struct timespec start, struct timespec end)
{
    return end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec) / 1000000000.0;
}

static double timeval_seconds(struct timeval value)
{
    return value.tv_sec + value.tv_usec / 1000000.0;
}

static uint64_t get_physical_offset(const char *path)
{
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0)
        return UINT64_MAX;

    size_t map_size = sizeof(struct fiemap) + sizeof(struct fiemap_extent);
    struct fiemap *map = calloc(1, map_size);
    if (!map) {
        close(fd);
        return UINT64_MAX;
    }

    map->fm_start = 0;
    map->fm_length = ~0ULL;
    map->fm_extent_count = 1;

    uint64_t physical = UINT64_MAX;

    if (ioctl(fd, FS_IOC_FIEMAP, map) == 0 && map->fm_mapped_extents)
        physical = map->fm_extents[0].fe_physical;

    free(map);
    close(fd);

    return physical;
}

static int collect_file(const char *path, const struct stat *stat_info, int type, struct FTW *ftw_info)
{
    (void)ftw_info;

    if (type != FTW_F || !S_ISREG(stat_info->st_mode))
        return 0;

    if (file_count >= file_limit) {
        collection_limit_reached = true;
        return 1;
    }

    if (file_count == file_capacity) {
        file_capacity = file_capacity ? file_capacity * 2 : 16384;
        struct FileEntry *new_files = realloc(files, file_capacity * sizeof(*files));
        if (!new_files)
            return 1;
        files = new_files;
    }

    files[file_count].path = strdup(path);
    if (!files[file_count].path)
        return 1;

    files[file_count].size = stat_info->st_size;
    files[file_count].physical = get_physical_offset(path);
    files[file_count].memory_offset = total_size;
    files[file_count].scan_index = file_count;

    total_size += stat_info->st_size;
    file_count++;

    if (file_count >= file_limit) {
        collection_limit_reached = true;
        return 1;
    }

    return 0;
}

static int compare_physical(const void *left, const void *right)
{
    const struct FileEntry *const *file_left = left;
    const struct FileEntry *const *file_right = right;

    if ((*file_left)->physical < (*file_right)->physical)
        return -1;

    if ((*file_left)->physical > (*file_right)->physical)
        return 1;

    if ((*file_left)->scan_index < (*file_right)->scan_index)
        return -1;

    if ((*file_left)->scan_index > (*file_right)->scan_index)
        return 1;

    return 0;
}

static int open_readonly(const char *path, int extra_flags)
{
    int fd = open(path, O_RDONLY | O_CLOEXEC | O_NOATIME | extra_flags);

    if (fd < 0 && errno == EPERM)
        fd = open(path, O_RDONLY | O_CLOEXEC | extra_flags);

    return fd;
}

static void print_progress(size_t completed, size_t count, size_t bytes_read, struct timespec start, struct timespec *last_update)
{
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);

    if (elapsed_seconds(*last_update, now) < 1.0 && completed != count)
        return;

    double seconds = elapsed_seconds(start, now);
    double percentage = count ? completed * 100.0 / count : 100.0;
    double mib = bytes_read / 1024.0 / 1024.0;
    double throughput = seconds > 0.0 ? mib / seconds : 0.0;

    printf("\r%6.2f%% | %zu/%zu files | %.1f MiB | %.1fs | %.1f MiB/s", percentage, completed, count, mib, seconds, throughput);
    fflush(stdout);
    *last_update = now;
}

static void evict_page_cache(struct FileEntry **order, size_t count)
{
    for (size_t i = 0; i < count; i++) {
        int fd = open(order[i]->path, O_RDONLY | O_CLOEXEC);
        if (fd < 0)
            continue;

        posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        close(fd);
    }

    struct timespec pause_time = {.tv_sec = 1, .tv_nsec = 0};
    nanosleep(&pause_time, NULL);
}

static struct DeviceStats read_device_stats(void)
{
    struct DeviceStats stats = {0};
    FILE *file = fopen("/proc/diskstats", "r");
    if (!file)
        return stats;

    unsigned target_major = major(source_device);
    unsigned target_minor = minor(source_device);
    char line[1024];

    while (fgets(line, sizeof(line), file)) {
        unsigned device_major;
        unsigned device_minor;
        char name[128];
        unsigned long long read_ios;
        unsigned long long read_merges;
        unsigned long long read_sectors;
        unsigned long long read_ticks;
        unsigned long long write_ios;
        unsigned long long write_merges;
        unsigned long long write_sectors;
        unsigned long long write_ticks;
        unsigned long long in_flight;
        unsigned long long io_ticks;
        unsigned long long weighted_ticks;

        int fields = sscanf(line, "%u %u %127s %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu",
            &device_major, &device_minor, name, &read_ios, &read_merges, &read_sectors, &read_ticks,
            &write_ios, &write_merges, &write_sectors, &write_ticks, &in_flight, &io_ticks, &weighted_ticks);

        if (fields < 14 || device_major != target_major || device_minor != target_minor)
            continue;

        stats.available = true;
        stats.read_ios = read_ios;
        stats.read_sectors = read_sectors;
        stats.read_ticks = read_ticks;
        stats.io_ticks = io_ticks;
        break;
    }

    fclose(file);
    return stats;
}

static void start_result(struct rusage *usage, struct timespec *start, struct DeviceStats *device_stats)
{
    getrusage(RUSAGE_SELF, usage);
    *device_stats = read_device_stats();
    clock_gettime(CLOCK_MONOTONIC, start);
}

static void finish_result(struct BenchmarkResult *result, struct rusage usage_start, struct timespec start, struct DeviceStats device_start)
{
    struct timespec end;
    struct rusage usage_end;
    struct DeviceStats device_end;

    clock_gettime(CLOCK_MONOTONIC, &end);
    getrusage(RUSAGE_SELF, &usage_end);
    device_end = read_device_stats();

    result->seconds = elapsed_seconds(start, end);
    result->mib_per_second = result->seconds > 0.0 ? result->bytes_read / 1024.0 / 1024.0 / result->seconds : 0.0;
    result->user_seconds = timeval_seconds(usage_end.ru_utime) - timeval_seconds(usage_start.ru_utime);
    result->system_seconds = timeval_seconds(usage_end.ru_stime) - timeval_seconds(usage_start.ru_stime);
    result->minor_faults = usage_end.ru_minflt - usage_start.ru_minflt;
    result->major_faults = usage_end.ru_majflt - usage_start.ru_majflt;

    if (device_start.available && device_end.available) {
        result->device_stats_available = true;
        result->device_read_bytes = (device_end.read_sectors - device_start.read_sectors) * 512ULL;
        result->device_read_ticks = device_end.read_ticks - device_start.read_ticks;
        result->device_io_ticks = device_end.io_ticks - device_start.io_ticks;
    }
}

static struct BenchmarkResult benchmark_buffered(const char *name, struct FileEntry **order, size_t count, char *memory, bool use_fadvise, bool use_dontcache)
{
    struct BenchmarkResult result = {.name = name, .supported = true};
    struct rusage usage_start;
    struct timespec start;
    struct timespec last_update;
    struct DeviceStats device_start;
    size_t bytes_read = 0;

    evict_page_cache(order, count);
    start_result(&usage_start, &start, &device_start);
    last_update = start;

    for (size_t i = 0; i < count; i++) {
        struct FileEntry *entry = order[i];
        int fd = open_readonly(entry->path, 0);
        if (fd < 0) {
            snprintf(result.note, sizeof(result.note), "open failed: %s", strerror(errno));
            result.supported = false;
            break;
        }

        if (use_fadvise) {
            posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
            posix_fadvise(fd, 0, 0, POSIX_FADV_NOREUSE);
        }

        size_t remaining = entry->size;
        size_t file_offset = 0;

        while (remaining) {
            ssize_t bytes;

            if (use_dontcache) {
                struct iovec vector = {
                    .iov_base = memory + entry->memory_offset + file_offset,
                    .iov_len = remaining,
                };
                bytes = preadv2(fd, &vector, 1, file_offset, RWF_DONTCACHE);
            } else {
                bytes = read(fd, memory + entry->memory_offset + file_offset, remaining);
            }

            if (bytes < 0 && use_dontcache && (errno == EOPNOTSUPP || errno == EINVAL || errno == ENOSYS)) {
                snprintf(result.note, sizeof(result.note), "RWF_DONTCACHE unsupported by this kernel/filesystem");
                result.supported = false;
                remaining = 0;
                break;
            }

            if (bytes <= 0) {
                if (result.supported)
                    snprintf(result.note, sizeof(result.note), "read failed: %s", bytes == 0 ? "unexpected EOF" : strerror(errno));
                result.supported = false;
                remaining = 0;
                break;
            }

            file_offset += bytes;
            remaining -= bytes;
            bytes_read += bytes;
        }

        close(fd);

        if (!result.supported)
            break;

        result.files_read = i + 1;
        print_progress(i + 1, count, bytes_read, start, &last_update);
    }

    result.bytes_read = bytes_read;
    finish_result(&result, usage_start, start, device_start);
    printf("\n");
    return result;
}

static struct BenchmarkResult benchmark_mmap(struct FileEntry **order, size_t count, char *memory)
{
    struct BenchmarkResult result = {.name = "mmap + MADV_SEQUENTIAL, physical order", .supported = true};
    struct rusage usage_start;
    struct timespec start;
    struct timespec last_update;
    struct DeviceStats device_start;
    size_t bytes_read = 0;

    evict_page_cache(order, count);
    start_result(&usage_start, &start, &device_start);
    last_update = start;

    for (size_t i = 0; i < count; i++) {
        struct FileEntry *entry = order[i];
        int fd = open_readonly(entry->path, 0);
        if (fd < 0) {
            snprintf(result.note, sizeof(result.note), "open failed: %s", strerror(errno));
            result.supported = false;
            break;
        }

        if (entry->size) {
            void *mapping = mmap(NULL, entry->size, PROT_READ, MAP_PRIVATE, fd, 0);
            if (mapping == MAP_FAILED) {
                snprintf(result.note, sizeof(result.note), "mmap failed: %s", strerror(errno));
                close(fd);
                result.supported = false;
                break;
            }

            madvise(mapping, entry->size, MADV_SEQUENTIAL);
            memcpy(memory + entry->memory_offset, mapping, entry->size);
            madvise(mapping, entry->size, MADV_DONTNEED);
            munmap(mapping, entry->size);
            bytes_read += entry->size;
        }

        close(fd);
        result.files_read = i + 1;
        print_progress(i + 1, count, bytes_read, start, &last_update);
    }

    result.bytes_read = bytes_read;
    finish_result(&result, usage_start, start, device_start);
    printf("\n");
    return result;
}

static size_t get_direct_alignment(const char *path, bool *from_statx)
{
    struct statx status = {0};
    *from_statx = false;

    if (statx(AT_FDCWD, path, AT_STATX_SYNC_AS_STAT, STATX_DIOALIGN, &status) == 0 &&
        (status.stx_mask & STATX_DIOALIGN) && status.stx_dio_mem_align && status.stx_dio_offset_align) {
        *from_statx = true;
        return status.stx_dio_mem_align > status.stx_dio_offset_align ? status.stx_dio_mem_align : status.stx_dio_offset_align;
    }

    return 4096;
}

static size_t align_up(size_t value, size_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

static bool create_direct_arena(struct DirectArena *arena, struct FileEntry **order, size_t count)
{
    memset(arena, 0, sizeof(*arena));
    arena->alignment = get_direct_alignment(order[0]->path, &arena->alignment_from_statx);
    arena->offsets = calloc(count, sizeof(*arena->offsets));
    if (!arena->offsets)
        return false;

    for (size_t i = 0; i < count; i++) {
        arena->offsets[i] = arena->size;
        size_t slot_size = align_up(order[i]->size ? order[i]->size : arena->alignment, arena->alignment);
        arena->size += slot_size;
    }

    void *memory = NULL;
    if (posix_memalign(&memory, arena->alignment, arena->size) != 0) {
        free(arena->offsets);
        arena->offsets = NULL;
        return false;
    }

    arena->memory = memory;
    memset(arena->memory, 0, arena->size);
    return true;
}

static void destroy_direct_arena(struct DirectArena *arena)
{
    free(arena->memory);
    free(arena->offsets);
    memset(arena, 0, sizeof(*arena));
}

static struct BenchmarkResult benchmark_direct(struct FileEntry **order, size_t count, struct DirectArena *arena)
{
    struct BenchmarkResult result = {.name = "O_DIRECT, physical order", .supported = true};
    struct rusage usage_start;
    struct timespec start;
    struct timespec last_update;
    struct DeviceStats device_start;
    size_t bytes_read = 0;

    evict_page_cache(order, count);
    start_result(&usage_start, &start, &device_start);
    last_update = start;

    for (size_t i = 0; i < count; i++) {
        struct FileEntry *entry = order[i];
        size_t request_size = align_up(entry->size, arena->alignment);
        int fd = open_readonly(entry->path, O_DIRECT);

        if (fd < 0) {
            snprintf(result.note, sizeof(result.note), "O_DIRECT open failed: %s", strerror(errno));
            result.supported = false;
            break;
        }

        if (entry->size) {
            ssize_t bytes = pread(fd, arena->memory + arena->offsets[i], request_size, 0);

            if (bytes < 0 && (errno == EINVAL || errno == EOPNOTSUPP)) {
                snprintf(result.note, sizeof(result.note), "O_DIRECT unsupported/alignment rejected (%zu-byte alignment%s)", arena->alignment,
                    arena->alignment_from_statx ? " from statx" : " fallback");
                close(fd);
                result.supported = false;
                break;
            }

            if (bytes != (ssize_t)entry->size) {
                snprintf(result.note, sizeof(result.note), "O_DIRECT short read: expected %zu, got %zd", entry->size, bytes);
                close(fd);
                result.supported = false;
                break;
            }

            bytes_read += bytes;
        }

        close(fd);
        result.files_read = i + 1;
        print_progress(i + 1, count, bytes_read, start, &last_update);
    }

    result.bytes_read = bytes_read;
    finish_result(&result, usage_start, start, device_start);

    if (result.supported)
        snprintf(result.note, sizeof(result.note), "%zu-byte alignment%s", arena->alignment,
            arena->alignment_from_statx ? " from statx" : " fallback");

    printf("\n");
    return result;
}

static int io_uring_setup_raw(unsigned entries, struct io_uring_params *params)
{
    return syscall(__NR_io_uring_setup, entries, params);
}

static int io_uring_enter_raw(int fd, unsigned to_submit, unsigned min_complete, unsigned flags)
{
    return syscall(__NR_io_uring_enter, fd, to_submit, min_complete, flags, NULL, 0);
}

static bool io_uring_init(struct IoUring *ring, unsigned entries, char *error_text, size_t error_size)
{
    memset(ring, 0, sizeof(*ring));
    ring->fd = -1;

    struct io_uring_params params = {0};
    int fd = io_uring_setup_raw(entries, &params);
    if (fd < 0) {
        snprintf(error_text, error_size, "io_uring_setup failed: %s", strerror(errno));
        return false;
    }

    ring->fd = fd;
    ring->sq_entries_count = params.sq_entries;
    ring->sq_ring_size = params.sq_off.array + params.sq_entries * sizeof(unsigned);
    ring->cq_ring_size = params.cq_off.cqes + params.cq_entries * sizeof(struct io_uring_cqe);

    if (params.features & IORING_FEAT_SINGLE_MMAP) {
        size_t ring_size = ring->sq_ring_size > ring->cq_ring_size ? ring->sq_ring_size : ring->cq_ring_size;
        ring->sq_ring_ptr = mmap(NULL, ring_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_SQ_RING);
        ring->cq_ring_ptr = ring->sq_ring_ptr;
    } else {
        ring->sq_ring_ptr = mmap(NULL, ring->sq_ring_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_SQ_RING);
        ring->cq_ring_ptr = mmap(NULL, ring->cq_ring_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_CQ_RING);
    }

    if (ring->sq_ring_ptr == MAP_FAILED || ring->cq_ring_ptr == MAP_FAILED) {
        snprintf(error_text, error_size, "io_uring ring mmap failed: %s", strerror(errno));
        return false;
    }

    ring->sqes = mmap(NULL, params.sq_entries * sizeof(struct io_uring_sqe), PROT_READ | PROT_WRITE,
        MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_SQES);

    if (ring->sqes == MAP_FAILED) {
        snprintf(error_text, error_size, "io_uring SQE mmap failed: %s", strerror(errno));
        return false;
    }

    ring->sq_head = (unsigned *)((char *)ring->sq_ring_ptr + params.sq_off.head);
    ring->sq_tail = (unsigned *)((char *)ring->sq_ring_ptr + params.sq_off.tail);
    ring->sq_ring_mask = (unsigned *)((char *)ring->sq_ring_ptr + params.sq_off.ring_mask);
    ring->sq_ring_entries = (unsigned *)((char *)ring->sq_ring_ptr + params.sq_off.ring_entries);
    ring->sq_flags = (unsigned *)((char *)ring->sq_ring_ptr + params.sq_off.flags);
    ring->sq_dropped = (unsigned *)((char *)ring->sq_ring_ptr + params.sq_off.dropped);
    ring->sq_array = (unsigned *)((char *)ring->sq_ring_ptr + params.sq_off.array);

    ring->cq_head = (unsigned *)((char *)ring->cq_ring_ptr + params.cq_off.head);
    ring->cq_tail = (unsigned *)((char *)ring->cq_ring_ptr + params.cq_off.tail);
    ring->cq_ring_mask = (unsigned *)((char *)ring->cq_ring_ptr + params.cq_off.ring_mask);
    ring->cq_ring_entries = (unsigned *)((char *)ring->cq_ring_ptr + params.cq_off.ring_entries);
    ring->cq_overflow = (unsigned *)((char *)ring->cq_ring_ptr + params.cq_off.overflow);
    ring->cqes = (struct io_uring_cqe *)((char *)ring->cq_ring_ptr + params.cq_off.cqes);

    return true;
}

static void io_uring_destroy(struct IoUring *ring)
{
    if (ring->sqes && ring->sqes != MAP_FAILED)
        munmap(ring->sqes, ring->sq_entries_count * sizeof(struct io_uring_sqe));

    if (ring->sq_ring_ptr && ring->sq_ring_ptr != MAP_FAILED) {
        if (ring->cq_ring_ptr == ring->sq_ring_ptr) {
            size_t size = ring->sq_ring_size > ring->cq_ring_size ? ring->sq_ring_size : ring->cq_ring_size;
            munmap(ring->sq_ring_ptr, size);
        } else {
            munmap(ring->sq_ring_ptr, ring->sq_ring_size);
            if (ring->cq_ring_ptr && ring->cq_ring_ptr != MAP_FAILED)
                munmap(ring->cq_ring_ptr, ring->cq_ring_size);
        }
    }

    if (ring->fd >= 0)
        close(ring->fd);
}

static bool io_uring_queue_read(struct IoUring *ring, struct IoJob *job, struct FileEntry *entry, char *memory)
{
    unsigned head = __atomic_load_n(ring->sq_head, __ATOMIC_ACQUIRE);
    unsigned tail = __atomic_load_n(ring->sq_tail, __ATOMIC_RELAXED);

    if (tail - head >= *ring->sq_ring_entries)
        return false;

    unsigned index = tail & *ring->sq_ring_mask;
    struct io_uring_sqe *sqe = &ring->sqes[index];
    memset(sqe, 0, sizeof(*sqe));

    size_t remaining = entry->size - job->completed;
    if (remaining > UINT_MAX)
        remaining = UINT_MAX;

    sqe->opcode = IORING_OP_READ;
    sqe->fd = job->fd;
    sqe->off = job->completed;
    sqe->addr = (uint64_t)(uintptr_t)(memory + entry->memory_offset + job->completed);
    sqe->len = remaining;
    sqe->user_data = job->file_index;

    ring->sq_array[index] = index;
    __atomic_store_n(ring->sq_tail, tail + 1, __ATOMIC_RELEASE);
    return true;
}

static struct BenchmarkResult benchmark_io_uring(struct FileEntry **order, size_t count, char *memory, unsigned queue_depth)
{
    char name[128];
    snprintf(name, sizeof(name), "io_uring buffered, physical order, QD %u", queue_depth);

    struct BenchmarkResult result = {.supported = true};
    char *name_copy = strdup(name);
    result.name = name_copy ? name_copy : "io_uring buffered";

    struct IoUring ring;
    char error_text[256] = {0};
    unsigned entries = queue_depth < IO_URING_ENTRIES ? IO_URING_ENTRIES : queue_depth * 2;

    if (!io_uring_init(&ring, entries, error_text, sizeof(error_text))) {
        result.supported = false;
        snprintf(result.note, sizeof(result.note), "%s", error_text);
        io_uring_destroy(&ring);
        return result;
    }

    struct IoJob *jobs = calloc(count, sizeof(*jobs));
    if (!jobs) {
        result.supported = false;
        snprintf(result.note, sizeof(result.note), "job allocation failed");
        io_uring_destroy(&ring);
        return result;
    }

    for (size_t i = 0; i < count; i++)
        jobs[i].fd = -1;

    evict_page_cache(order, count);

    struct rusage usage_start;
    struct timespec start;
    struct timespec last_update;
    struct DeviceStats device_start;
    size_t next_file = 0;
    size_t completed_files = 0;
    size_t bytes_read = 0;
    unsigned in_flight = 0;
    unsigned pending_submissions = 0;

    start_result(&usage_start, &start, &device_start);
    last_update = start;

    while (next_file < count && in_flight < queue_depth) {
        struct IoJob *job = &jobs[next_file];
        job->file_index = next_file;
        job->fd = open_readonly(order[next_file]->path, 0);

        if (job->fd < 0 || !io_uring_queue_read(&ring, job, order[next_file], memory)) {
            snprintf(result.note, sizeof(result.note), "initial io_uring submission failed: %s", job->fd < 0 ? strerror(errno) : "SQ full");
            result.supported = false;
            break;
        }

        next_file++;
        in_flight++;
        pending_submissions++;
    }

    if (result.supported && pending_submissions) {
        if (io_uring_enter_raw(ring.fd, pending_submissions, 1, IORING_ENTER_GETEVENTS) < 0) {
            snprintf(result.note, sizeof(result.note), "io_uring_enter failed: %s", strerror(errno));
            result.supported = false;
        }
        pending_submissions = 0;
    }

    while (result.supported && completed_files < count) {
        unsigned cq_head = __atomic_load_n(ring.cq_head, __ATOMIC_ACQUIRE);
        unsigned cq_tail = __atomic_load_n(ring.cq_tail, __ATOMIC_ACQUIRE);

        if (cq_head == cq_tail) {
            if (io_uring_enter_raw(ring.fd, pending_submissions, 1, IORING_ENTER_GETEVENTS) < 0) {
                snprintf(result.note, sizeof(result.note), "io_uring wait failed: %s", strerror(errno));
                result.supported = false;
                break;
            }
            pending_submissions = 0;
            continue;
        }

        while (cq_head != cq_tail) {
            struct io_uring_cqe *cqe = &ring.cqes[cq_head & *ring.cq_ring_mask];
            size_t index = cqe->user_data;
            struct IoJob *job = &jobs[index];
            struct FileEntry *entry = order[index];

            if (cqe->res < 0) {
                snprintf(result.note, sizeof(result.note), "io_uring read failed: %s", strerror(-cqe->res));
                result.supported = false;
                break;
            }

            if (cqe->res == 0 && job->completed < entry->size) {
                snprintf(result.note, sizeof(result.note), "io_uring unexpected EOF");
                result.supported = false;
                break;
            }

            job->completed += cqe->res;
            bytes_read += cqe->res;

            if (job->completed < entry->size) {
                if (!io_uring_queue_read(&ring, job, entry, memory)) {
                    snprintf(result.note, sizeof(result.note), "io_uring SQ full during short-read resubmission");
                    result.supported = false;
                    break;
                }
                pending_submissions++;
            } else {
                close(job->fd);
                job->fd = -1;
                completed_files++;
                in_flight--;

                if (next_file < count) {
                    struct IoJob *next_job = &jobs[next_file];
                    next_job->file_index = next_file;
                    next_job->fd = open_readonly(order[next_file]->path, 0);

                    if (next_job->fd < 0 || !io_uring_queue_read(&ring, next_job, order[next_file], memory)) {
                        snprintf(result.note, sizeof(result.note), "io_uring replacement submission failed: %s",
                            next_job->fd < 0 ? strerror(errno) : "SQ full");
                        result.supported = false;
                        break;
                    }

                    next_file++;
                    in_flight++;
                    pending_submissions++;
                }
            }

            cq_head++;
            __atomic_store_n(ring.cq_head, cq_head, __ATOMIC_RELEASE);
            cq_tail = __atomic_load_n(ring.cq_tail, __ATOMIC_ACQUIRE);
        }

        if (!result.supported)
            break;

        result.files_read = completed_files;
        print_progress(completed_files, count, bytes_read, start, &last_update);

        if (pending_submissions) {
            if (io_uring_enter_raw(ring.fd, pending_submissions, 0, 0) < 0) {
                snprintf(result.note, sizeof(result.note), "io_uring submit failed: %s", strerror(errno));
                result.supported = false;
                break;
            }
            pending_submissions = 0;
        }
    }

    result.bytes_read = bytes_read;
    finish_result(&result, usage_start, start, device_start);
    printf("\n");

    for (size_t i = 0; i < count; i++) {
        if (jobs[i].fd >= 0)
            close(jobs[i].fd);
    }

    free(jobs);
    io_uring_destroy(&ring);
    return result;
}

static bool io_uring_queue_direct(struct IoUring *ring, struct IoJob *job, struct FileEntry *entry, struct DirectArena *arena)
{
    unsigned head = __atomic_load_n(ring->sq_head, __ATOMIC_ACQUIRE);
    unsigned tail = __atomic_load_n(ring->sq_tail, __ATOMIC_RELAXED);

    if (tail - head >= *ring->sq_ring_entries)
        return false;

    unsigned index = tail & *ring->sq_ring_mask;
    struct io_uring_sqe *sqe = &ring->sqes[index];
    memset(sqe, 0, sizeof(*sqe));

    size_t request_size = align_up(entry->size, arena->alignment);
    if (request_size > UINT_MAX)
        return false;

    sqe->opcode = IORING_OP_READ;
    sqe->fd = job->fd;
    sqe->off = 0;
    sqe->addr = (uint64_t)(uintptr_t)(arena->memory + arena->offsets[job->file_index]);
    sqe->len = request_size;
    sqe->user_data = job->file_index;

    ring->sq_array[index] = index;
    __atomic_store_n(ring->sq_tail, tail + 1, __ATOMIC_RELEASE);
    return true;
}

static struct BenchmarkResult benchmark_io_uring_direct(struct FileEntry **order, size_t count, struct DirectArena *arena, unsigned queue_depth)
{
    char name[128];
    snprintf(name, sizeof(name), "io_uring O_DIRECT, physical order, QD %u", queue_depth);

    struct BenchmarkResult result = {.supported = true};
    char *name_copy = strdup(name);
    result.name = name_copy ? name_copy : "io_uring O_DIRECT";

    struct IoUring ring;
    char error_text[256] = {0};
    unsigned entries = queue_depth < IO_URING_ENTRIES ? IO_URING_ENTRIES : queue_depth * 2;

    if (!io_uring_init(&ring, entries, error_text, sizeof(error_text))) {
        result.supported = false;
        snprintf(result.note, sizeof(result.note), "%s", error_text);
        io_uring_destroy(&ring);
        return result;
    }

    struct IoJob *jobs = calloc(count, sizeof(*jobs));
    if (!jobs) {
        result.supported = false;
        snprintf(result.note, sizeof(result.note), "job allocation failed");
        io_uring_destroy(&ring);
        return result;
    }

    for (size_t i = 0; i < count; i++)
        jobs[i].fd = -1;

    evict_page_cache(order, count);

    struct rusage usage_start;
    struct timespec start;
    struct timespec last_update;
    struct DeviceStats device_start;
    size_t next_file = 0;
    size_t completed_files = 0;
    size_t bytes_read = 0;
    unsigned in_flight = 0;
    unsigned pending_submissions = 0;

    start_result(&usage_start, &start, &device_start);
    last_update = start;

    while (next_file < count && in_flight < queue_depth) {
        struct IoJob *job = &jobs[next_file];
        job->file_index = next_file;
        job->fd = open_readonly(order[next_file]->path, O_DIRECT);

        if (job->fd < 0 || !io_uring_queue_direct(&ring, job, order[next_file], arena)) {
            snprintf(result.note, sizeof(result.note), "initial O_DIRECT io_uring submission failed: %s",
                job->fd < 0 ? strerror(errno) : "SQ full or request too large");
            result.supported = false;
            break;
        }

        next_file++;
        in_flight++;
        pending_submissions++;
    }

    if (result.supported && pending_submissions) {
        if (io_uring_enter_raw(ring.fd, pending_submissions, 1, IORING_ENTER_GETEVENTS) < 0) {
            snprintf(result.note, sizeof(result.note), "io_uring_enter failed: %s", strerror(errno));
            result.supported = false;
        }
        pending_submissions = 0;
    }

    while (result.supported && completed_files < count) {
        unsigned cq_head = __atomic_load_n(ring.cq_head, __ATOMIC_ACQUIRE);
        unsigned cq_tail = __atomic_load_n(ring.cq_tail, __ATOMIC_ACQUIRE);

        if (cq_head == cq_tail) {
            if (io_uring_enter_raw(ring.fd, pending_submissions, 1, IORING_ENTER_GETEVENTS) < 0) {
                snprintf(result.note, sizeof(result.note), "io_uring wait failed: %s", strerror(errno));
                result.supported = false;
                break;
            }
            pending_submissions = 0;
            continue;
        }

        while (cq_head != cq_tail) {
            struct io_uring_cqe *cqe = &ring.cqes[cq_head & *ring.cq_ring_mask];
            size_t index = cqe->user_data;
            struct IoJob *job = &jobs[index];
            struct FileEntry *entry = order[index];

            if (cqe->res < 0) {
                snprintf(result.note, sizeof(result.note), "O_DIRECT io_uring read failed: %s", strerror(-cqe->res));
                result.supported = false;
                break;
            }

            if (cqe->res != (int)entry->size) {
                snprintf(result.note, sizeof(result.note), "O_DIRECT io_uring short read: expected %zu, got %d", entry->size, cqe->res);
                result.supported = false;
                break;
            }

            bytes_read += cqe->res;
            close(job->fd);
            job->fd = -1;
            completed_files++;
            in_flight--;

            if (next_file < count) {
                struct IoJob *next_job = &jobs[next_file];
                next_job->file_index = next_file;
                next_job->fd = open_readonly(order[next_file]->path, O_DIRECT);

                if (next_job->fd < 0 || !io_uring_queue_direct(&ring, next_job, order[next_file], arena)) {
                    snprintf(result.note, sizeof(result.note), "O_DIRECT io_uring replacement submission failed: %s",
                        next_job->fd < 0 ? strerror(errno) : "SQ full or request too large");
                    result.supported = false;
                    break;
                }

                next_file++;
                in_flight++;
                pending_submissions++;
            }

            cq_head++;
            __atomic_store_n(ring.cq_head, cq_head, __ATOMIC_RELEASE);
            cq_tail = __atomic_load_n(ring.cq_tail, __ATOMIC_ACQUIRE);
        }

        if (!result.supported)
            break;

        result.files_read = completed_files;
        print_progress(completed_files, count, bytes_read, start, &last_update);

        if (pending_submissions) {
            if (io_uring_enter_raw(ring.fd, pending_submissions, 0, 0) < 0) {
                snprintf(result.note, sizeof(result.note), "O_DIRECT io_uring submit failed: %s", strerror(errno));
                result.supported = false;
                break;
            }
            pending_submissions = 0;
        }
    }

    result.bytes_read = bytes_read;
    finish_result(&result, usage_start, start, device_start);

    if (result.supported)
        snprintf(result.note, sizeof(result.note), "%zu-byte alignment%s", arena->alignment,
            arena->alignment_from_statx ? " from statx" : " fallback");

    printf("\n");

    for (size_t i = 0; i < count; i++) {
        if (jobs[i].fd >= 0)
            close(jobs[i].fd);
    }

    free(jobs);
    io_uring_destroy(&ring);
    return result;
}

static void write_sysfs_value(FILE *report, const char *device_path, const char *name)
{
    char path[PATH_MAX];
    snprintf(path, sizeof(path), "%s/queue/%s", device_path, name);

    FILE *file = fopen(path, "r");
    if (!file)
        return;

    char value[512];
    if (fgets(value, sizeof(value), file)) {
        value[strcspn(value, "\r\n")] = '\0';
        fprintf(report, "%s: %s\n", name, value);
    }

    fclose(file);
}

static bool find_queue_device_path(char *output, size_t output_size)
{
    char link_path[PATH_MAX];
    char resolved[PATH_MAX];
    snprintf(link_path, sizeof(link_path), "/sys/dev/block/%u:%u", major(source_device), minor(source_device));

    if (!realpath(link_path, resolved))
        return false;

    char current[PATH_MAX];
    snprintf(current, sizeof(current), "%s", resolved);

    for (int depth = 0; depth < 6; depth++) {
        char queue_path[PATH_MAX];
        const char *queue_suffix = "/queue/rotational";
        if (strlen(current) + strlen(queue_suffix) + 1 >= sizeof(queue_path))
            break;
        strcpy(queue_path, current);
        strcat(queue_path, queue_suffix);

        if (access(queue_path, R_OK) == 0) {
            snprintf(output, output_size, "%s", current);
            return true;
        }

        char *slash = strrchr(current, '/');
        if (!slash || slash == current)
            break;
        *slash = '\0';
    }

    return false;
}

static void write_environment(FILE *report, const char *directory)
{
    struct utsname kernel;
    uname(&kernel);

    fprintf(report, "Linux file-read benchmark report\n");
    fprintf(report, "================================\n\n");
    fprintf(report, "Directory: %s\n", directory);
    fprintf(report, "Kernel: %s %s %s\n", kernel.sysname, kernel.release, kernel.machine);
    fprintf(report, "Device: %u:%u\n", major(source_device), minor(source_device));
    fprintf(report, "Page size: %ld bytes\n", sysconf(_SC_PAGESIZE));
    fprintf(report, "Files selected: %zu\n", file_count);
    fprintf(report, "Directory scan + FIEMAP setup: %.3f s\n", scan_seconds);
    fprintf(report, "Selected compressed bytes: %zu (%.1f MiB)\n", total_size, total_size / 1024.0 / 1024.0);

    size_t mapped_files = 0;
    for (size_t i = 0; i < file_count; i++) {
        if (files[i].physical != UINT64_MAX)
            mapped_files++;
    }
    fprintf(report, "FIEMAP physical offsets available: %zu/%zu\n", mapped_files, file_count);

    char device_path[PATH_MAX];
    if (find_queue_device_path(device_path, sizeof(device_path))) {
        fprintf(report, "Block queue path: %s\n", device_path);
        write_sysfs_value(report, device_path, "rotational");
        write_sysfs_value(report, device_path, "scheduler");
        write_sysfs_value(report, device_path, "read_ahead_kb");
        write_sysfs_value(report, device_path, "max_sectors_kb");
        write_sysfs_value(report, device_path, "max_hw_sectors_kb");
        write_sysfs_value(report, device_path, "nomerges");
        write_sysfs_value(report, device_path, "nr_requests");
    }

    fprintf(report, "\nCache handling: POSIX_FADV_DONTNEED is issued for every selected file before each timed run.\n");
    fprintf(report, "Memory handling: the shared destination arena is prefaulted before benchmarking so anonymous page faults do not favor later tests.\n\n");
}

static void write_result(FILE *report, struct BenchmarkResult *result)
{
    fprintf(report, "%s\n", result->name);
    fprintf(report, "  Supported: %s\n", result->supported ? "yes" : "no");

    if (result->supported) {
        fprintf(report, "  Files: %zu\n", result->files_read);
        fprintf(report, "  Bytes: %zu (%.1f MiB)\n", result->bytes_read, result->bytes_read / 1024.0 / 1024.0);
        fprintf(report, "  Wall time: %.3f s\n", result->seconds);
        fprintf(report, "  Throughput: %.2f MiB/s\n", result->mib_per_second);
        fprintf(report, "  User CPU: %.3f s\n", result->user_seconds);
        fprintf(report, "  System CPU: %.3f s\n", result->system_seconds);
        fprintf(report, "  Minor faults: %ld\n", result->minor_faults);
        fprintf(report, "  Major faults: %ld\n", result->major_faults);

        if (result->device_stats_available) {
            fprintf(report, "  /proc/diskstats bytes read: %.1f MiB\n", result->device_read_bytes / 1024.0 / 1024.0);
            fprintf(report, "  /proc/diskstats read time: %.3f s\n", result->device_read_ticks / 1000.0);
            fprintf(report, "  /proc/diskstats device busy time: %.3f s\n", result->device_io_ticks / 1000.0);

            if (result->bytes_read && result->device_read_bytes < result->bytes_read * 8 / 10)
                fprintf(report, "  Cache warning: diskstats reports less than 80%% of requested bytes reaching this block device; treat this run as cache-contaminated or verify layered-device accounting.\n");
        }
    }

    if (result->note[0])
        fprintf(report, "  Note: %s\n", result->note);

    fprintf(report, "\n");
}

static int compare_result_speed(const void *left, const void *right)
{
    const struct BenchmarkResult *result_left = left;
    const struct BenchmarkResult *result_right = right;

    if (!result_left->supported && result_right->supported)
        return 1;
    if (result_left->supported && !result_right->supported)
        return -1;
    if (result_left->mib_per_second > result_right->mib_per_second)
        return -1;
    if (result_left->mib_per_second < result_right->mib_per_second)
        return 1;
    return 0;
}

static void write_summary(FILE *report, struct BenchmarkResult *results, size_t result_count)
{
    struct BenchmarkResult *sorted = malloc(result_count * sizeof(*sorted));
    if (!sorted)
        return;

    memcpy(sorted, results, result_count * sizeof(*sorted));
    qsort(sorted, result_count, sizeof(*sorted), compare_result_speed);

    fprintf(report, "Ranking by measured throughput\n");
    fprintf(report, "==============================\n\n");

    size_t rank = 1;
    for (size_t i = 0; i < result_count; i++) {
        if (!sorted[i].supported)
            continue;
        fprintf(report, "%zu. %.2f MiB/s - %s\n", rank++, sorted[i].mib_per_second, sorted[i].name);
    }

    fprintf(report, "\nUnsupported tests\n");
    fprintf(report, "-----------------\n");
    bool any_unsupported = false;

    for (size_t i = 0; i < result_count; i++) {
        if (sorted[i].supported)
            continue;
        any_unsupported = true;
        fprintf(report, "- %s: %s\n", sorted[i].name, sorted[i].note[0] ? sorted[i].note : "unsupported");
    }

    if (!any_unsupported)
        fprintf(report, "None\n");

    free(sorted);
}

static void free_files(void)
{
    for (size_t i = 0; i < file_count; i++)
        free(files[i].path);
    free(files);
}

int main(int argc, char **argv)
{
    if (argc < 2 || argc > 4) {
        fprintf(stderr, "usage: %s DIRECTORY [FILE_LIMIT] [REPORT_PATH]\n", argv[0]);
        return 1;
    }

    if (argc >= 3) {
        char *end = NULL;
        unsigned long long requested_limit = strtoull(argv[2], &end, 10);
        if (!requested_limit || !end || *end) {
            fprintf(stderr, "invalid FILE_LIMIT: %s\n", argv[2]);
            return 1;
        }
        file_limit = requested_limit;
    }

    const char *report_path = argc >= 4 ? argv[3] : REPORT_PATH;

    struct stat directory_status;
    if (stat(argv[1], &directory_status) != 0) {
        perror(argv[1]);
        return 1;
    }
    source_device = directory_status.st_dev;

    printf("Scanning up to %zu files and collecting FIEMAP positions...\n", file_limit);
    struct timespec scan_start;
    struct timespec scan_end;
    clock_gettime(CLOCK_MONOTONIC, &scan_start);
    int walk_result = nftw(argv[1], collect_file, 64, FTW_PHYS);
    clock_gettime(CLOCK_MONOTONIC, &scan_end);
    scan_seconds = elapsed_seconds(scan_start, scan_end);

    if (walk_result != 0 && !collection_limit_reached) {
        fprintf(stderr, "directory scan failed\n");
        free_files();
        return 1;
    }

    if (!file_count) {
        fprintf(stderr, "no files found\n");
        free_files();
        return 1;
    }

    printf("Selected %zu files, %.1f MiB compressed\n", file_count, total_size / 1024.0 / 1024.0);

    struct FileEntry **scan_order = malloc(file_count * sizeof(*scan_order));
    struct FileEntry **physical_order = malloc(file_count * sizeof(*physical_order));
    if (!scan_order || !physical_order) {
        fprintf(stderr, "could not allocate ordering arrays\n");
        free(scan_order);
        free(physical_order);
        free_files();
        return 1;
    }

    for (size_t i = 0; i < file_count; i++) {
        scan_order[i] = &files[i];
        physical_order[i] = &files[i];
    }

    qsort(physical_order, file_count, sizeof(*physical_order), compare_physical);

    char *memory = mmap(NULL, total_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (memory == MAP_FAILED) {
        perror("mmap destination arena");
        free(scan_order);
        free(physical_order);
        free_files();
        return 1;
    }

    printf("Prefaulting %.1f MiB destination arena...\n", total_size / 1024.0 / 1024.0);
    memset(memory, 0, total_size);

    struct DirectArena direct_arena;
    if (!create_direct_arena(&direct_arena, physical_order, file_count)) {
        fprintf(stderr, "could not allocate O_DIRECT benchmark arena\n");
        munmap(memory, total_size);
        free(scan_order);
        free(physical_order);
        free_files();
        return 1;
    }

    struct BenchmarkResult results[16];
    size_t result_count = 0;

    printf("\n[1/16] Buffered read, filesystem scan order\n");
    results[result_count++] = benchmark_buffered("buffered read, filesystem scan order", scan_order, file_count, memory, false, false);

    printf("\n[2/16] Buffered read, physical order\n");
    results[result_count++] = benchmark_buffered("buffered read, physical order", physical_order, file_count, memory, false, false);

    printf("\n[3/16] Buffered read + POSIX_FADV_SEQUENTIAL/NOREUSE, physical order\n");
    results[result_count++] = benchmark_buffered("buffered + FADV_SEQUENTIAL/NOREUSE, physical order", physical_order, file_count, memory, true, false);

    printf("\n[4/16] preadv2 + RWF_DONTCACHE + fadvise, physical order\n");
    results[result_count++] = benchmark_buffered("preadv2 RWF_DONTCACHE + fadvise, physical order", physical_order, file_count, memory, true, true);

    printf("\n[5/16] mmap + MADV_SEQUENTIAL, physical order\n");
    results[result_count++] = benchmark_mmap(physical_order, file_count, memory);

    printf("\n[6/16] O_DIRECT, physical order\n");
    results[result_count++] = benchmark_direct(physical_order, file_count, &direct_arena);

    unsigned queue_depths[] = {1, 2, 4, 8, 16};
    for (size_t i = 0; i < sizeof(queue_depths) / sizeof(queue_depths[0]); i++) {
        printf("\n[%zu/16] io_uring buffered, physical order, queue depth %u\n", 7 + i, queue_depths[i]);
        results[result_count++] = benchmark_io_uring(physical_order, file_count, memory, queue_depths[i]);
    }

    for (size_t i = 0; i < sizeof(queue_depths) / sizeof(queue_depths[0]); i++) {
        printf("\n[%zu/16] io_uring O_DIRECT, physical order, queue depth %u\n", 12 + i, queue_depths[i]);
        results[result_count++] = benchmark_io_uring_direct(physical_order, file_count, &direct_arena, queue_depths[i]);
    }

    FILE *report = fopen(report_path, "w");
    if (!report) {
        perror(report_path);
    } else {
        write_environment(report, argv[1]);
        fprintf(report, "Individual results\n");
        fprintf(report, "==================\n\n");

        for (size_t i = 0; i < result_count; i++)
            write_result(report, &results[i]);

        write_summary(report, results, result_count);
        fclose(report);
        printf("\nReport written to %s\n", report_path);
    }

    for (size_t i = 0; i < result_count; i++) {
        if (strncmp(results[i].name, "io_uring buffered,", 18) == 0 || strncmp(results[i].name, "io_uring O_DIRECT,", 18) == 0)
            free((void *)results[i].name);
    }

    destroy_direct_arena(&direct_arena);
    munmap(memory, total_size);
    free(scan_order);
    free(physical_order);
    free_files();
    return report ? 0 : 1;
}
