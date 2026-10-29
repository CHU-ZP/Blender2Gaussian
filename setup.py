from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='mesh_to_gs',
    version='0.1.0',
    packages=['mesh_to_gs'],
    ext_modules=[
        CUDAExtension(
            '_C',  
            sources=[
                'csrc/bindings.cpp',
                'csrc/mesh_to_gs_kernel.cu',
            ],
            # BuildExtension selects the C++ ABI to match the installed torch.
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
