"""
游戏内GPU计算与通用计算系统 - 验收测试
运行方式：python -m pytest tests/test_gpu_compute.py -q
共 16 个测试用例
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpu.compute import GPUCompute


class TestGPUCompute:
    def setup_method(self):
        self.system = GPUCompute({"workgroup_size": 4, "memory_total": 1024})

    # ---- 1. 计算调度：工作组 ----
    def test_case_01_dispatch_workgroups(self):
        result = self.system.dispatch_compute("square", [1, 2, 3, 4, 5, 6, 7, 8, 9])
        assert result["device"] == "gpu"
        assert result["workgroups"] == 3  # 9 个元素，每组 4 个
        assert result["result"] == [1, 4, 9, 16, 25, 36, 49, 64, 81]

    def test_case_02_dispatch_explicit_workgroups(self):
        result = self.system.dispatch_compute(lambda x: x + 1,
                                              [0, 1, 2, 3], workgroups=2)
        assert result["workgroups"] == 2
        assert result["result"] == [1, 2, 3, 4]

    def test_case_03_dispatch_sync_default(self):
        # wait=True：调度返回时计算已同步完成
        result = self.system.dispatch_compute("double", [1, 2, 3])
        assert result["synced"] is True
        assert result["result"] == [2, 4, 6]

    def test_case_04_dispatch_async_then_synchronize(self):
        # wait=False：先返回未完成任务，synchronize() 等待完成
        job = self.system.dispatch_compute("square", [1, 2, 3, 4], wait=False)
        assert job["synced"] is False
        assert job["result"] is None
        sync = self.system.synchronize(job["job"])
        assert sync["synced"] is True
        assert sync["results"][job["job"]] == [1, 4, 9, 16]

    def test_case_05_dispatch_invalid_kernel(self):
        with pytest.raises(ValueError):
            self.system.dispatch_compute("not_a_kernel", [1, 2, 3])

    # ---- 2. 通用计算：缓冲区 + 传输 ----
    def test_case_06_general_purpose_runs_on_compute_pipeline(self):
        result = self.system.general_purpose([1, 2, 3], "square")
        assert result["pipeline"] == "compute"  # 不经过图形管线
        assert result["result"] == [1, 4, 9]

    def test_case_07_general_purpose_uses_buffer(self):
        result = self.system.general_purpose([2, 3], "double")
        assert isinstance(result["buffer"], int)

    def test_case_08_buffer_transfer_roundtrip(self):
        buffer_id = self.system.create_buffer()
        up = self.system.transfer_to_gpu(buffer_id, [5, 6, 7])
        assert up["direction"] == "cpu_to_gpu" and up["on_gpu"] is True
        down = self.system.transfer_to_cpu(buffer_id)
        assert down["direction"] == "gpu_to_cpu"
        assert down["data"] == [5, 6, 7]

    def test_case_09_general_purpose_transfer_counts(self):
        result = self.system.general_purpose([1, 2, 3, 4], "double")
        assert result["transferred_to_gpu"] == 4
        assert result["transferred_to_cpu"] == 4

    # ---- 3. 并行计算：映射 + 归约 ----
    def test_case_10_parallel_map(self):
        result = self.system.parallel_processing([1, 2, 3, 4], lambda x: x * x)
        assert result["threads"] == 4
        assert result["mapped"] == [1, 4, 9, 16]

    def test_case_11_parallel_map_to_threads(self):
        data = list(range(100))
        result = self.system.parallel_processing(data, lambda x: x + 1, reduce=None)
        assert result["threads"] == 100
        assert result["mapped"] == list(range(1, 101))
        assert result["reduced"] is None

    def test_case_12_parallel_reduce_builtins(self):
        result = self.system.parallel_processing([1, 2, 3, 4], lambda x: x, reduce="sum")
        assert result["reduced"] == 10
        result = self.system.parallel_processing([1, 2, 3, 4], lambda x: x, reduce="max")
        assert result["reduced"] == 4

    def test_case_13_parallel_reduce_custom(self):
        result = self.system.parallel_processing(
            [1, 2, 3, 4], lambda x: x, reduce=lambda a, b: a * b)
        assert result["reduced"] == 24

    # ---- 4. 计算着色器：语法 + 执行 ----
    def test_case_14_compute_shader_syntax_and_execution(self):
        source = """
        #version 450
        layout(local_size_x = 64) in;
        void main() {
            float x = x_in;
            out = x * x + 1;
        }
        """
        result = self.system.compute_shader(source, [1, 2, 3])
        assert result["compiled"] is True
        assert result["executed"] is True
        assert result["local_size"] == 64
        assert result["result"] == [2, 5, 10]

    def test_case_15_compute_shader_bad_syntax(self):
        with pytest.raises(SyntaxError):
            self.system.compute_shader("void foo() {", [1, 2])
        with pytest.raises(SyntaxError):
            self.system.compute_shader("out = x *;", [1, 2])

    # ---- 5. GPU内存管理：分配 + 释放 ----
    def test_case_16_memory_allocate_free_and_stats(self):
        before = self.system.gpu_memory("stats")
        assert before["used"] == 0 and before["free"] == 1024

        alloc = self.system.gpu_memory("allocate", size=256)
        assert alloc["allocated"] is True
        handle = alloc["handle"]

        stats = self.system.gpu_memory("stats")
        assert stats["used"] == 256 and stats["free"] == 768 and stats["allocations"] == 1

        with pytest.raises(MemoryError):
            self.system.gpu_memory("allocate", size=2048)  # 超过显存

        freed = self.system.gpu_memory("free", handle=handle)
        assert freed["freed"] is True
        assert self.system.gpu_memory("stats")["used"] == 0

        with pytest.raises(RuntimeError):
            self.system.gpu_memory("free", handle=handle)  # 不能重复释放


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
