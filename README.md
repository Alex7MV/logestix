# logestix

An enterprise-grade industrial inference system designed for high-performance AI workloads.

## Building

### Prerequisites

- CMake 3.15+
- C++17/C11 compiler (GCC, Clang, MSVC)
- NVIDIA CCCL (added as submodule)

### Clone with submodules

```bash
git clone --recurse-submodules https://github.com/<your-org>/logestix.git
```

If you already cloned without `--recurse-submodules`, run:

```bash
git submodule update --init --recursive
```

### Configure & build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

The executable `lsxfabric` will be placed in `build/src/`.

---

## Licensing

This project is dual-licensed to accommodate both open-source development and commercial production environments.

### Open Source License
The open-source version of **logestix** is licensed under the [GNU Affero General Public License v3.0 (AGPL-3.0)](./LICENSE). 

* **Please note:** AGPL-3.0 is a strong copyleft license. If you modify this software or integrate it into a hosted platform or cloud service (SaaS), you are legally required to make your entire corporate infrastructure/source code interacting with this system open-source under the same license.

### Commercial License (Enterprise)
For commercial deployments, enterprise infrastructure, closed-source integration, or hosted services, the AGPL-3.0 limitations can be removed. 

We offer a **Commercial Enterprise License** that includes:
* Exemption from AGPL-3.0 source-code disclosure requirements.
* Professional Technical Support & Service Level Agreements (SLA).
* Custom integrations and hardware-specific optimizations.

To obtain a commercial license or request a private evaluation copy (PoC), please contact us at: **<your-sales-email@example.com>**

---

## Contributing

We welcome community contributions! However, to ensure we can protect our dual-licensing model, all contributors are required to sign our **Contributor License Agreement (CLA)** before any Pull Request can be merged. 

The CLA signing process is automated via GitHub as soon as you open a Pull Request.
