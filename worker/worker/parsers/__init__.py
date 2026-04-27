# Parser imports trigger registration via the @register decorator.
# Add new parser modules here as they are implemented.
from . import apps  # noqa: F401
from . import attachments  # noqa: F401
from . import calendar  # noqa: F401
from . import files  # noqa: F401
from . import gmail  # noqa: F401
from . import obsidian  # noqa: F401
from . import omnifocus  # noqa: F401
from . import repositories  # noqa: F401
from . import slack  # noqa: F401
