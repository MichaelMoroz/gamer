# GAMER
Galaxy Ray Tracer in Qt

# Setting up **vcpkg** (inside `build/`), installing **Qt 5**, and using it from CMake

These instructions assume **Windows 10/11**, Visual Studio 2022 with the *Desktop C++* workload, and CMake ≥ 3.25 in your `PATH`.

---

## 1. Prepare a clean working tree

```powershell
git clone <your-repo-url> gamer
cd gamer
git submodule update --init --recursive   # if you use submodules
```

---

## 2. Put **vcpkg** inside the project’s `build/` directory

```powershell
mkdir build
git clone https://github.com/microsoft/vcpkg.git build\vcpkg
build\vcpkg\bootstrap-vcpkg.bat
```

*Result:* the toolchain file will be at  
`build\vcpkg\scripts\buildsystems\vcpkg.cmake`.

---

## 3. Install Qt 5 for the **x64-windows** triplet

```powershell
build\vcpkg\vcpkg install qt5 --triplet x64-windows
```

> **Note:** The first install compiles Qt (5.15) and all dependencies; it takes time.  
> Packages land in `build\vcpkg\installed\x64-windows\`.

Optional—but helpful for MSBuild projects:

```powershell
build\vcpkg\vcpkg integrate install
```

---

## 4. Configure and build with CMake

Always pass the toolchain file *and* the triplet:

```powershell
cmake -S . -B build\x64 -DCMAKE_TOOLCHAIN_FILE=%CD%\build\vcpkg\scripts\buildsystems\vcpkg.cmake -DVCPKG_TARGET_TRIPLET=x64-windows -A x64
cmake --build build\x64 --config Release
```

