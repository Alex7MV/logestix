#include "lsxfabric/core.h"
#include "lsxfabric/version.h"

static int initialized = 0;

int core_init(void)
{
    initialized = 1;
    return 0;
}

const char* core_version(void)
{
    return LSKFABRIC_VERSION_STRING;
}
