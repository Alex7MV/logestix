#include "lsxfabric/core.h"
#include "lsxfabric/version.h"

#include <cstdio>
#include <cstdlib>

int main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;

    if (core_init() != 0) {
        std::fprintf(stderr, "core_init failed\n");
        return EXIT_FAILURE;
    }

    std::printf("lsxfabric v%s\n", core_version());
    return EXIT_SUCCESS;
}
