import os
os.environ['DISTRIBUTED_TESTS_DEFAULT_TIMEOUT'] = '30000'

import torch
import os
try:
    from torch.testing._internal.common_distributed import DistributedTestBase
except ImportError:
    from torch.testing._internal.common_distributed import MultiProcessTestCase
    class DistributedTestBase(MultiProcessTestCase):

        def setUp(self):
            super().setUp()
            os.environ["WORLD_SIZE"] = str(self.world_size)
            self._spawn_processes()

        def tearDown(self):
            try:
                torch.distributed.destroy_process_group()
            except AssertionError:
                pass
            try:
                os.remove(self.file_name)
            except OSError:
                pass

        def backend(self, device) -> str:
            if "cuda" in device:
                return "nccl"
            elif "hpu" in device:  # intel gaudi
                return "hccl"
            elif "xpu" in device:
                return "xccl"
            else:
                return "gloo"

        def create_pg(self, device, world_size=None):
            if world_size is None:
                world_size = self.world_size
            num_visible_devices = torch.get_device_module(device).device_count()
            store = torch.distributed.FileStore(self.file_name, num_visible_devices)
            torch.distributed.init_process_group(
                backend=self.backend(device),
                world_size=world_size,
                rank=self.rank,
                store=store,
            )
            if "nccl" in self.backend(device) or "xccl" in self.backend(device):
                torch.accelerator.set_device_index(self.rank)
            return torch.distributed.distributed_c10d._get_default_group()

        def rank_to_device(self, device):
            num_visible_devices = torch.get_device_module(device).device_count()
            return {i: [i % num_visible_devices] for i in range(self.world_size)}

from torch import distributed as dist

from hy_parallelism.parallel_states import init_parallel_state, get_parallel_state, get_or_init_parallel_state

import loguru
from loguru import logger
def with_world_size(world_size: int):
    """Decorator to specify world_size for a test method."""
    def decorator(func):
        func._test_world_size = world_size
        return func
    return decorator


class DistributedTest(DistributedTestBase):
    master_addr = 'localhost'
    def dump_grad(self, module, full_tensor=True, clone=True):
        dic = {}
        for name, param in module.named_parameters():
            grad = param.grad
            if grad is None:
                continue
            if hasattr(grad, 'full_tensor') and full_tensor:
                grad = grad.full_tensor()
            dic[name] = grad.clone() if clone else grad
        return dic

    @property
    def world_size(self) -> int:
        """Override world_size to support per-test customization."""
        # Get current test method name
        test_name = self._current_test_name()
        # Get the test method
        test_method = getattr(self, test_name, None)
        # Check if the test method has a custom world_size
        if test_method and hasattr(test_method, '_test_world_size'):
            return test_method._test_world_size
        # Default to parent's world_size
        return super().world_size

    @property
    def master_port(self) -> int:
        """
        Find and return an available TCP port to use as the MASTER_PORT for distributed training.
        This ensures that concurrent test runs don't collide.
        """
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        port = s.getsockname()[1]
        s.close()
        return port


    # @property
    # def world_size(self) -> int:
    #     return 8

    def rank0_log(self, *args, **kwargs):
        if dist.get_rank() == 0:
            loguru.logger.info(*args, **kwargs)

    def setup_environment_variables(self):
        os.environ['WORLD_SIZE'] = str(self.world_size)
        os.environ['RANK'] = str(dist.get_rank())
        os.environ['LOCAL_RANK'] = str(dist.get_rank() % 8)
        os.environ['MASTER_ADDR'] = self.master_addr
        os.environ['MASTER_PORT'] = str(self.master_port)

    def configure_logger(self):
        pass
        # loguru.logger.remove(None)
        # loguru.logger.add(
        #     sys.stdout,
        #     format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level:^8}</level> | <level><bold>[Rank "
        #     + os.environ.get('RANK', '0')
        #     + "]: {message}</bold></level> (<cyan>{file}:{line}</cyan>)",
        # )

    def create_pg(self, device, world_size=None):
        if world_size is None:
            world_size = self.world_size
        num_visible_devices = torch.get_device_module(device).device_count()
        store = torch.distributed.FileStore(self.file_name, num_visible_devices)
        torch.distributed.init_process_group(
            backend=self.backend(device),
            world_size=world_size,
            rank=self.rank,
            store=store,
        )
        # if "nccl" in self.backend(device) or "xccl" in self.backend(device):
        #     torch.accelerator.set_device_index(self.rank)
        return torch.distributed.distributed_c10d._get_default_group()

    def init_process_group(self, device='cuda'):
        if self.world_size > 8:
            device = 'cpu' # use gloo
        # old_rank = self.rank
        local_rank = self.rank % torch.cuda.device_count()
        # self.rank = local_rank
        self.create_pg(device)
        # self.rank = old_rank


        self.setup_environment_variables()
        self.configure_logger()
        torch.cuda.set_device(local_rank)

    def set_seed(self, seed=0):
        # from accelerate.utils import set_seed
        # set_seed(seed)
        from hy_parallelism.utils import set_determinism
        set_determinism(None, torch.device('cuda'), seed=seed)

    def set_proper_seed(self, seed=0, gcd_dp_size=None):
        from hy_parallelism.parallel_states import get_parallel_state
        parallel_dims = get_parallel_state()
        dp_rank = parallel_dims.dp_mesh.get_local_rank()
        if gcd_dp_size is not None:
            dp_rank = dp_rank % gcd_dp_size
        self.set_seed(seed + dp_rank)

    def init_parallel_state(self, *args, **kwargs):
        init_parallel_state(*args, **kwargs)

    def get_or_init_parallel_state(self, *args, **kwargs):
        return get_or_init_parallel_state(*args, **kwargs)

    def get_parallel_state(self, *args, **kwargs):
        return get_parallel_state(*args, **kwargs)


    def setUp(self):
        super().setUp()


class DistributedTest8Proc(DistributedTest):
    @property
    def world_size(self) -> int:
        return 8

class DistributedTest4Proc(DistributedTest):
    @property
    def world_size(self) -> int:
        return 4


class DistributedTest2Proc(DistributedTest):
    @property
    def world_size(self) -> int:
        return 2

class DistributedTest1Proc(DistributedTest):
    @property
    def world_size(self) -> int:
        return 1

class DistributedTest32Proc(DistributedTest):
    @property
    def world_size(self) -> int:
        return 32

class DistributedTest64Proc(DistributedTest):
    @property
    def world_size(self) -> int:
        return 64

class DistributedTest16Proc(DistributedTest):
    @property
    def world_size(self) -> int:
        return 16
    


from torch.testing._internal.common_utils import parametrize, instantiate_parametrized_tests

def my_parametrize(keys, values):
    keys = keys.replace(' ', '')
    keys = keys.split(',')
    for sample in values:
        yield {key:value for key, value in zip(keys, sample)}

