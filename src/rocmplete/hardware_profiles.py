"""Canonical ROCmplete hardware profile and architecture identities."""

PROFILE_ARCHITECTURES = (
    ("rdna4", ("gfx1200", "gfx1201")),
    ("strix-halo", ("gfx1151",)),
    ("strix-point", ("gfx1150",)),
)

GPU_PROFILES = tuple(profile for profile, _ in PROFILE_ARCHITECTURES)
SUPPORTED_ARCHITECTURES = tuple(
    sorted(
        architecture
        for _, architectures in PROFILE_ARCHITECTURES
        for architecture in architectures
    )
)
ARCHITECTURE_PROFILES = {
    architecture: profile
    for profile, architectures in PROFILE_ARCHITECTURES
    for architecture in architectures
}
PROFILES = ("auto",) + GPU_PROFILES + ("cpu",)
