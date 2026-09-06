from geometry.incremental_triangulator import (
    IncrementalTriangulator,
    TriangulatorOptions,
)

class ValidCamera:
    def __init__(self):
        self.calls = 0

    def HasBogusParams(
        self,
        min_focal_length_ratio,
        max_focal_length_ratio,
        max_extra_param,
    ):
        self.calls += 1
        return False


class InvalidCamera:
    def __init__(self):
        self.calls = 0

    def HasBogusParams(
        self,
        min_focal_length_ratio,
        max_focal_length_ratio,
        max_extra_param,
    ):
        self.calls += 1
        return True


valid_camera = ValidCamera()
invalid_camera = InvalidCamera()

triangulator = IncrementalTriangulator(
    TriangulatorOptions()
)

v1 = triangulator.IsCameraValid(
    1,
    valid_camera,
)

v2 = triangulator.IsCameraValid(
    1,
    valid_camera,
)

bad = triangulator.IsCameraValid(
    2,
    invalid_camera,
)

bad2 = triangulator.IsCameraValid(
    2,
    invalid_camera,
)

print("VALID FIRST:", v1)
print("VALID SECOND:", v2)
print("VALID CAMERA CALLS:", valid_camera.calls)

print("INVALID FIRST:", bad)
print("INVALID SECOND:", bad2)
print("INVALID CAMERA CALLS:", invalid_camera.calls)

print("CACHE:", triangulator.camera_validity_cache)

passed = (
    v1 is True
    and v2 is True
    and bad is False
    and bad2 is False
    and valid_camera.calls == 1
    and invalid_camera.calls == 1
    and triangulator.camera_validity_cache == {
        1: True,
        2: False,
    }
)

print(
    "CAMERA VALIDATION TEST: PASS"
    if passed
    else "CAMERA VALIDATION TEST: FAIL"
)
