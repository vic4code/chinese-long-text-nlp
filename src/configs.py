from pathlib import Path

from dotenv import load_dotenv

import judge

PROJECT_BASE = Path(judge.__file__).parent
DOTENV_FILE_PATH = PROJECT_BASE / '.env.prod'

load_dotenv(DOTENV_FILE_PATH)
