from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='extensions',
    ext_modules=[
        CUDAExtension('HT_opp', [
            'cuda_opps/HT.cpp',
            'cuda_opps/HT_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })