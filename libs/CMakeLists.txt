lsx_warning_policy()

# libs/lsxstorage must be defined before libs/lsxcommon: the artifact reader sits
# BELOW the model layer so ModelBase::BeginLoad can use it without a cycle.
add_subdirectory(lsxstorage)
add_subdirectory(lsxcommon)
add_subdirectory(lsxnetwork)
