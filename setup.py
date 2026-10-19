"""Installation unique des paquets MADEVO (`devo`, `utils`, `ms_model`).

Les trois extensions C++/CUDA héritées de DPVO (`cuda_corr`, `cuda_ba`,
`lietorch_backends`) sont compilées par défaut. Poser ``MADEVO_SKIP_CUDA_EXT=1``
les saute, pour une installation CPU locale :

    MADEVO_SKIP_CUDA_EXT=1 pip install -e . --no-build-isolation
"""
import os
import os.path as osp
import shutil
import sys

from setuptools import find_packages, setup

ROOT = osp.dirname(osp.abspath(__file__))


def _truthy(name):
    return os.environ.get(name, "").strip().lower() not in ("", "0", "false", "no")


def find_eigen():
    """Répertoire contenant les en-têtes Eigen (Eigen/Dense), ou None."""
    candidates = [
        os.environ.get("EIGEN_INCLUDE_DIR"),
        osp.join(ROOT, "thirdparty", "eigen-3.4.0"),
    ]
    prefix = os.environ.get("CONDA_PREFIX")
    if prefix:
        candidates += [osp.join(prefix, "include", "eigen3"), osp.join(prefix, "include")]
    candidates += ["/usr/include/eigen3", "/usr/local/include/eigen3"]
    for path in candidates:
        if path and osp.isfile(osp.join(path, "Eigen", "Dense")):
            return path
    return None


def cuda_extensions():
    """Les extensions à compiler, ou [] si on les saute explicitement."""
    if _truthy("MADEVO_SKIP_CUDA_EXT"):
        print("MADEVO: MADEVO_SKIP_CUDA_EXT posé, extensions CUDA non compilées "
              "(DEVO ne tournera pas dans cet env).")
        return []

    if shutil.which("nvcc") is None:
        sys.exit(
            "MADEVO: nvcc introuvable, impossible de compiler les extensions CUDA.\n"
            "  - sur un serveur GPU : créer l'env avec environment.yml "
            "(mamba env create -f environment.yml), qui fournit cuda-nvcc 11.8 ;\n"
            "  - en local sans GPU : réinstaller avec MADEVO_SKIP_CUDA_EXT=1."
        )

    eigen = find_eigen()
    if eigen is None:
        sys.exit(
            "MADEVO: en-têtes Eigen introuvables (Eigen/Dense).\n"
            "  bash scripts/download/download_eigen.sh\n"
            "ou poser EIGEN_INCLUDE_DIR vers un Eigen 3.4 existant."
        )
    print(f"MADEVO: Eigen trouvé dans {eigen}")

    from torch.utils.cpp_extension import CUDAExtension

    args = {"cxx": ["-O3"], "nvcc": ["-O3"]}
    return [
        CUDAExtension(
            "cuda_corr",
            sources=["devo/altcorr/correlation.cpp", "devo/altcorr/correlation_kernel.cu"],
            extra_compile_args=args,
        ),
        CUDAExtension(
            "cuda_ba",
            sources=["devo/fastba/ba.cpp", "devo/fastba/ba_cuda.cu"],
            extra_compile_args=args,
        ),
        CUDAExtension(
            "lietorch_backends",
            include_dirs=[osp.join(ROOT, "devo/lietorch/include"), eigen],
            sources=[
                "devo/lietorch/src/lietorch.cpp",
                "devo/lietorch/src/lietorch_gpu.cu",
                "devo/lietorch/src/lietorch_cpu.cpp",
            ],
            extra_compile_args=args,
        ),
    ]


def packages_and_dirs():
    """(packages, package_dir) pour devo, utils et ms_model.

    `ms_model` est encore sous MS_Model/ ; la remontée à la racine (tâche T3)
    n'oblige pas à toucher ce fichier.
    """
    packages = find_packages(where=ROOT, include=["devo*"])
    package_dir = {}

    # `utils` et ses modules n'ont pas d'__init__.py (paquet implicite).
    packages.append("utils")

    if osp.isdir(osp.join(ROOT, "ms_model")):
        packages += find_packages(where=ROOT, include=["ms_model*"])
    elif osp.isdir(osp.join(ROOT, "MS_Model", "ms_model")):
        ms_packages = find_packages(where=osp.join(ROOT, "MS_Model"), include=["ms_model*"])
        packages += ms_packages
        for pkg in ms_packages:
            package_dir[pkg] = osp.join("MS_Model", *pkg.split("."))
    else:
        sys.exit("MADEVO: paquet ms_model introuvable (ni ./ms_model ni ./MS_Model/ms_model)")

    return packages, package_dir


packages, package_dir = packages_and_dirs()
ext_modules = cuda_extensions()
cmdclass = {}
if ext_modules:
    from torch.utils.cpp_extension import BuildExtension

    cmdclass["build_ext"] = BuildExtension

setup(
    packages=packages,
    package_dir=package_dir,
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
