"""TidyBot2 control package.

Keep package import lightweight so nodes that do not require Interbotix
dependencies (for example, simulation-only nodes) can still start.
"""

__all__: list[str] = []

try:
    from tidybot_control.gripper_controller import GripperController
    __all__.append('GripperController')
except Exception:
    # Optional dependency path for real-hardware gripper control.
    pass
