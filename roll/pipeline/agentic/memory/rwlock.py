import asyncio
from typing import Optional


class AsyncRWLock:
    """
    Async Read-Write Lock with writer priority.

    - Searches acquire read lock (can run concurrently)
    - Updates acquire write lock (exclusive, blocks all searches)
    - Prevents race conditions where search reads UIDs that get deleted mid-search
    """

    def __init__(self):
        self._readers = 0  # Number of active readers
        self._writers = 0  # Number of active writers (0 or 1)
        self._waiting_writers = 0  # Number of writers waiting
        self._read_ready = asyncio.Condition()  # Condition for readers
        self._write_ready = asyncio.Condition()  # Condition for writers

    async def acquire_read(self):
        """
        Acquire read lock.

        Blocks if:
        - A writer is active
        - Writers are waiting (writer priority)
        """
        async with self._read_ready:
            # Wait while writer is active or writers are waiting
            while self._writers > 0 or self._waiting_writers > 0:
                await self._read_ready.wait()

            self._readers += 1

    async def release_read(self):
        """
        Release read lock.

        Notifies waiting writers if this was the last reader.
        """
        should_notify_writer = False
        async with self._read_ready:
            self._readers -= 1

            # If no more readers, we need to wake up a waiting writer
            # But we can't acquire _write_ready while holding _read_ready (deadlock risk)
            # So we'll release _read_ready first, then notify
            if self._readers == 0:
                should_notify_writer = True

        # Notify writer after releasing read lock to avoid deadlock
        if should_notify_writer:
            async with self._write_ready:
                self._write_ready.notify()

    async def acquire_write(self):
        """
        Acquire write lock (exclusive).

        Blocks if:
        - Any readers are active
        - Another writer is active
        """
        async with self._write_ready:
            self._waiting_writers += 1

            # Wait while readers or writers are active
            while self._readers > 0 or self._writers > 0:
                await self._write_ready.wait()

            self._waiting_writers -= 1
            self._writers += 1

    async def release_write(self):
        """
        Release write lock.

        Notifies all waiting readers and one waiting writer.
        """
        async with self._write_ready:
            self._writers -= 1

            # Wake up waiting writers first (writer priority)
            self._write_ready.notify()

            # Then wake up all waiting readers
            async with self._read_ready:
                self._read_ready.notify_all()

    def reader(self):
        """
        Context manager for read lock.

        Usage:
            async with rwlock.reader():
                # Read operations
                data = memory.get_search_info()
        """
        return _ReaderContext(self)

    def writer(self):
        """
        Context manager for write lock.

        Usage:
            async with rwlock.writer():
                # Write operations
                memory.update(data)
        """
        return _WriterContext(self)


class _ReaderContext:
    """Context manager for read lock"""

    def __init__(self, rwlock: AsyncRWLock):
        self._rwlock = rwlock

    async def __aenter__(self):
        await self._rwlock.acquire_read()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._rwlock.release_read()
        return False


class _WriterContext:
    """Context manager for write lock"""

    def __init__(self, rwlock: AsyncRWLock):
        self._rwlock = rwlock

    async def __aenter__(self):
        await self._rwlock.acquire_write()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._rwlock.release_write()
        return False
