"""
游戏内GPU计算与通用计算系统 - 核心模块

在纯Python中模拟GPU设备：
- dispatch_compute:    计算调度（工作组划分 + 同步等待）
- general_purpose:     通用计算（GPU缓冲区 + CPU<->GPU数据传输）
- parallel_processing: 并行计算（线程映射 + 结果归约）
- compute_shader:      计算着色器（语法解析 + 执行）
- gpu_memory:          GPU内存管理（分配 + 释放）
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
import math
import re
import time


@dataclass
class Config:
    workgroup_size: int = 64
    memory_total: int = 1 << 30  # 1 GiB 模拟显存
    max_threads: int = 8


# 计算着色器允许使用的内置函数（安全求值环境）
_SHADER_GLOBALS = {
    "__builtins__": {},
    "abs": abs, "min": min, "max": max, "pow": pow, "round": round,
    "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "exp": math.exp, "log": math.log, "floor": math.floor, "ceil": math.ceil,
    "pi": math.pi, "e": math.e,
}

# 调度器内置内核
_BUILTIN_KERNELS: Dict[str, Callable[[float], float]] = {
    "identity": lambda x: x,
    "square": lambda x: x * x,
    "double": lambda x: 2 * x,
    "negate": lambda x: -x,
    "abs": abs,
    "sqrt": lambda x: math.sqrt(x),
}


class GPUCompute:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        defaults = Config()
        self.workgroup_size = int(self.config.get("workgroup_size", defaults.workgroup_size))
        self.memory_total = int(self.config.get("memory_total", defaults.memory_total))
        self.max_threads = int(self.config.get("max_threads", defaults.max_threads))

        self._state = {}
        self._history = []

        # GPU 内存管理：handle -> {"size": int, "freed": bool}
        self._memory: Dict[int, Dict] = {}
        self._memory_used = 0
        self._next_handle = 1

        # GPU 缓冲区：buffer_id -> {"data": list, "on_gpu": bool}
        self._buffers: Dict[int, Dict] = {}
        self._next_buffer = 1

        # 异步计算任务：job_id -> {"done": bool, "result": any}
        self._jobs: Dict[int, Dict] = {}
        self._next_job = 1

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def update(self, dt: float):
        """推进模拟设备一帧（保留接口）。"""
        self._state["time"] = self._state.get("time", 0.0) + dt

    def reset(self):
        self._state = {}
        self._history = []
        self._memory = {}
        self._memory_used = 0
        self._next_handle = 1
        self._buffers = {}
        self._next_buffer = 1
        self._jobs = {}
        self._next_job = 1

    # ------------------------------------------------------------------
    # 1. 计算调度：工作组 + 同步
    # ------------------------------------------------------------------
    def dispatch_compute(
        self,
        kernel: Union[str, Callable],
        data: List[float],
        workgroups: Optional[int] = None,
        wait: bool = True,
    ) -> Dict:
        """把数据划分成工作组，在GPU上调度执行内核。

        - kernel: 内置内核名或可调用对象
        - workgroups: 工作组数量（默认按 workgroup_size 自动划分）
        - wait: True 时同步等待计算完成；False 时返回未完成任务，需 synchronize()
        """
        if isinstance(kernel, str):
            if kernel not in _BUILTIN_KERNELS:
                raise ValueError(f"未知内置内核: {kernel}")
            func = _BUILTIN_KERNELS[kernel]
        elif callable(kernel):
            func = kernel
        else:
            raise TypeError("kernel 必须是内置内核名或可调用对象")

        items = list(data)
        if workgroups is None:
            workgroups = max(1, math.ceil(len(items) / self.workgroup_size)) if items else 1
        if workgroups < 1:
            raise ValueError("workgroups 必须 >= 1")

        # 把数据均匀划分到各工作组
        groups = [[] for _ in range(workgroups)]
        for index, value in enumerate(items):
            groups[index % workgroups].append(value)

        job_id = self._next_job
        self._next_job += 1
        job = {"done": False, "result": None, "workgroups": workgroups, "device": "gpu"}
        self._jobs[job_id] = job

        def run():
            # 每个工作组由GPU线程块并行处理
            with ThreadPoolExecutor(max_workers=min(self.max_threads, workgroups)) as pool:
                group_results = list(pool.map(lambda g: [func(v) for v in g], groups))
            # 按原始顺序还原
            result = [None] * len(items)
            positions = [iter(range(i, len(items), workgroups)) for i in range(workgroups)]
            for group_index, group_result in enumerate(group_results):
                for value in group_result:
                    result[next(positions[group_index])] = value
            return result

        if wait:
            result = run()
            job["done"] = True
            job["result"] = result
            self._history.append({"op": "dispatch_compute", "workgroups": workgroups,
                                  "elements": len(items), "synced": True})
            return {"device": "gpu", "job": job_id, "workgroups": workgroups,
                    "synced": True, "result": result}

        job["run"] = run
        self._history.append({"op": "dispatch_compute", "workgroups": workgroups,
                              "elements": len(items), "synced": False})
        return {"device": "gpu", "job": job_id, "workgroups": workgroups,
                "synced": False, "result": None}

    def synchronize(self, job: Optional[int] = None) -> Dict:
        """同步等待：阻塞直到指定任务（或全部任务）计算完成。"""
        job_ids = [job] if job is not None else list(self._jobs.keys())
        for job_id in job_ids:
            entry = self._jobs.get(job_id)
            if entry is None:
                raise KeyError(f"未知任务: {job_id}")
            if not entry["done"]:
                entry["result"] = entry.pop("run")()
                entry["done"] = True
        self._history.append({"op": "synchronize", "jobs": list(job_ids)})
        done = all(j["done"] for j in self._jobs.values())
        results = {jid: self._jobs[jid]["result"] for jid in job_ids}
        return {"synced": True, "all_done": done, "results": results}

    # ------------------------------------------------------------------
    # 2. 通用计算：缓冲区 + 传输
    # ------------------------------------------------------------------
    def create_buffer(self, data: Optional[List[float]] = None, size: int = 0) -> int:
        """在GPU上创建缓冲区，返回缓冲区句柄。"""
        buffer_id = self._next_buffer
        self._next_buffer += 1
        initial = list(data) if data is not None else [0.0] * size
        self._buffers[buffer_id] = {"data": initial, "on_gpu": False}
        self._history.append({"op": "create_buffer", "buffer": buffer_id,
                              "size": len(initial)})
        return buffer_id

    def transfer_to_gpu(self, buffer_id: int, data: Optional[List[float]] = None) -> Dict:
        """CPU -> GPU 数据传输。"""
        buf = self._require_buffer(buffer_id)
        if data is not None:
            buf["data"] = list(data)
        buf["on_gpu"] = True
        self._history.append({"op": "transfer_to_gpu", "buffer": buffer_id,
                              "bytes": len(buf["data"]) * 8})
        return {"buffer": buffer_id, "direction": "cpu_to_gpu",
                "elements": len(buf["data"]), "on_gpu": True}

    def transfer_to_cpu(self, buffer_id: int) -> Dict:
        """GPU -> CPU 数据传输，读回计算结果。"""
        buf = self._require_buffer(buffer_id)
        if not buf["on_gpu"]:
            raise RuntimeError("数据不在GPU上，无法读回")
        buf["on_gpu"] = False
        self._history.append({"op": "transfer_to_cpu", "buffer": buffer_id,
                              "bytes": len(buf["data"]) * 8})
        return {"buffer": buffer_id, "direction": "gpu_to_cpu",
                "data": list(buf["data"]), "on_gpu": False}

    def general_purpose(
        self,
        data: List[float],
        operation: Union[str, Callable] = "square",
    ) -> Dict:
        """通用计算（GPGPU）：不经过图形管线，直接在GPU缓冲区上计算。

        流程：创建缓冲区 -> CPU传入GPU -> GPU计算 -> 读回CPU。
        """
        if isinstance(operation, str):
            if operation not in _BUILTIN_KERNELS:
                raise ValueError(f"未知操作: {operation}")
            func = _BUILTIN_KERNELS[operation]
        elif callable(operation):
            func = operation
        else:
            raise TypeError("operation 必须是内置操作名或可调用对象")

        buffer_id = self.create_buffer()
        up = self.transfer_to_gpu(buffer_id, data)
        buf = self._buffers[buffer_id]
        buf["data"] = [func(v) for v in buf["data"]]  # GPU 上原地计算
        down = self.transfer_to_cpu(buffer_id)

        self._history.append({"op": "general_purpose", "elements": len(down["data"])})
        return {"device": "gpu", "pipeline": "compute",  # 非图形管线
                "buffer": buffer_id,
                "transferred_to_gpu": up["elements"],
                "transferred_to_cpu": len(down["data"]),
                "result": down["data"]}

    def _require_buffer(self, buffer_id: int) -> Dict:
        if buffer_id not in self._buffers:
            raise KeyError(f"未知缓冲区: {buffer_id}")
        return self._buffers[buffer_id]

    # ------------------------------------------------------------------
    # 3. 并行计算：映射 + 归约
    # ------------------------------------------------------------------
    def parallel_processing(
        self,
        data: List[float],
        func: Optional[Callable] = None,
        reduce: Union[str, Callable, None] = "sum",
    ) -> Dict:
        """并行计算：把每个元素映射到一个GPU线程，再归约各线程结果。"""
        items = list(data)
        map_func = func if func is not None else _BUILTIN_KERNELS["identity"]
        if not callable(map_func):
            raise TypeError("func 必须是可调用对象")

        # 映射：一个元素一个线程，真正并行执行
        with ThreadPoolExecutor(max_workers=self.max_threads) as pool:
            mapped = list(pool.map(map_func, items))

        # 归约：把多个线程的结果合并为单个值
        reduced = None
        if reduce is not None:
            reduced = self._reduce(mapped, reduce)

        self._history.append({"op": "parallel_processing", "threads": len(items),
                              "reduce": reduce if isinstance(reduce, str) else "custom"})
        return {"device": "gpu", "threads": len(items),
                "mapped": mapped, "reduced": reduced}

    @staticmethod
    def _reduce(values: List[float], reduce: Union[str, Callable]) -> float:
        if callable(reduce):
            result = values[0] if values else None
            for value in values[1:]:
                result = reduce(result, value)
            return result
        if not values:
            return 0
        if reduce == "sum":
            return sum(values)
        if reduce == "product":
            return math.prod(values)
        if reduce == "max":
            return max(values)
        if reduce == "min":
            return min(values)
        if reduce == "mean":
            return sum(values) / len(values)
        raise ValueError(f"未知归约方式: {reduce}")

    # ------------------------------------------------------------------
    # 4. 计算着色器：语法 + 执行
    # ------------------------------------------------------------------
    def compile_shader(self, source: str) -> Dict:
        """解析并校验计算着色器语法，返回编译产物。"""
        if not isinstance(source, str) or not source.strip():
            raise ValueError("着色器源码不能为空")

        # 语法检查：必须包含 main 入口，且括号/花括号配对
        if "main" not in source:
            raise SyntaxError("计算着色器缺少 main 入口")
        for open_ch, close_ch in (("{", "}"), ("(", ")")):
            if source.count(open_ch) != source.count(close_ch):
                raise SyntaxError(f"括号不配对: {open_ch}{close_ch}")

        # 提取局部工作组大小声明：layout(local_size_x = N) in;
        match = re.search(r"local_size_x\s*=\s*(\d+)", source)
        local_size = int(match.group(1)) if match else 1

        # 提取可执行表达式：`out = <表达式>;`，表达式可引用 x（当前元素）
        expr_match = re.search(r"\bout\s*=\s*([^;]+);", source)
        expression = expr_match.group(1).strip() if expr_match else None
        if expression is None:
            # 允许简写形式：整个源码就是一个表达式，如 "x * 2 + 1"
            if re.fullmatch(r"[0-9a-zA-Z_+\-*/%().,\s]+", source.strip()):
                expression = source.strip()
            else:
                raise SyntaxError("着色器缺少可执行语句: out = <表达式>;")
        # 校验表达式可编译
        compile(expression, "<compute_shader>", "eval")

        return {"compiled": True, "local_size": local_size,
                "expression": expression, "source": source}

    def compute_shader(
        self,
        source: str,
        data: Optional[List[float]] = None,
    ) -> Dict:
        """编译并在GPU上执行计算着色器。"""
        program = self.compile_shader(source)
        items = list(data) if data is not None else []

        expression = program["expression"]

        def run_element(x):
            return eval(compile(expression, "<compute_shader>", "eval"),
                        dict(_SHADER_GLOBALS), {"x": x})

        # 按 local_size 划分线程组并行执行
        results = []
        with ThreadPoolExecutor(max_workers=self.max_threads) as pool:
            results = list(pool.map(run_element, items))

        self._history.append({"op": "compute_shader",
                              "local_size": program["local_size"],
                              "elements": len(items)})
        return {"device": "gpu", "compiled": True, "executed": True,
                "local_size": program["local_size"], "result": results}

    # ------------------------------------------------------------------
    # 5. GPU内存管理：分配 + 释放
    # ------------------------------------------------------------------
    def gpu_memory(self, operation: str, **kwargs) -> Dict:
        """GPU内存管理入口。

        - operation="allocate", size=<字节>  -> 分配显存，返回 handle
        - operation="free",     handle=<句柄> -> 释放显存
        - operation="stats"                   -> 内存统计
        """
        if operation == "allocate":
            return self._allocate(int(kwargs.get("size", 0)))
        if operation == "free":
            return self._free(kwargs.get("handle"))
        if operation == "stats":
            return {"total": self.memory_total, "used": self._memory_used,
                    "free": self.memory_total - self._memory_used,
                    "allocations": sum(1 for m in self._memory.values()
                                       if not m["freed"])}
        raise ValueError(f"未知内存操作: {operation}")

    def _allocate(self, size: int) -> Dict:
        if size <= 0:
            raise ValueError("分配大小必须为正数")
        if self._memory_used + size > self.memory_total:
            raise MemoryError(
                f"GPU显存不足: 请求 {size} 字节, 剩余 {self.memory_total - self._memory_used} 字节")
        handle = self._next_handle
        self._next_handle += 1
        self._memory[handle] = {"size": size, "freed": False}
        self._memory_used += size
        self._history.append({"op": "allocate", "handle": handle, "size": size})
        return {"handle": handle, "size": size, "allocated": True,
                "used": self._memory_used}

    def _free(self, handle: Optional[int]) -> Dict:
        if handle not in self._memory:
            raise KeyError(f"未知内存句柄: {handle}")
        block = self._memory[handle]
        if block["freed"]:
            raise RuntimeError(f"内存已被释放: {handle}")
        block["freed"] = True
        self._memory_used -= block["size"]
        self._history.append({"op": "free", "handle": handle, "size": block["size"]})
        return {"handle": handle, "freed": True, "used": self._memory_used}
