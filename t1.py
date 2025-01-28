import os
import logging
from typing import Dict, List, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_env_var(name: str, value: str) -> List[Tuple[int, str, int]]:
    """
    Check an environment variable for non-ASCII characters.
    Returns list of (position, character, unicode_value) for any non-ASCII characters.
    """
    problems = []
    for i, char in enumerate(value):
        if ord(char) > 127:  # Non-ASCII character
            problems.append((i, char, ord(char)))
    return problems

def format_unicode_name(char: str) -> str:
    """Get the Unicode name for a character, with fallback."""
    try:
        return f"'{char}' (U+{ord(char):04X})"
    except Exception:
        return f"U+{ord(char):04X}"

def check_api_keys() -> Dict[str, List[Tuple[int, str, int]]]:
    """
    Check common API key environment variables for non-ASCII characters.
    Returns dict of variable names and their problems.
    """
    api_vars = [
        'OPENAI_API_KEY',
        'GROQ_API_KEY',
        'ANTHROPIC_API_KEY',
        'PALM_API_KEY',
        'COHERE_API_KEY',
        # Add more API keys as needed
    ]
    
    results = {}
    for var in api_vars:
        if var in os.environ:
            value = os.environ[var]
            problems = check_env_var(var, value)
            if problems:
                results[var] = problems
                
            # Log the check
            if problems:
                logger.warning(f"Found non-ASCII characters in {var}:")
                for pos, char, code in problems:
                    logger.warning(f"  Position {pos}: {format_unicode_name(char)}")
            else:
                logger.info(f"✓ {var} contains only ASCII characters")
    
    return results

def main():
    logger.info("Checking API key environment variables for non-ASCII characters...")
    
    results = check_api_keys()
    
    if results:
        print("\nProblematic environment variables found:")
        print("========================================")
        for var_name, problems in results.items():
            print(f"\n{var_name}:")
            for pos, char, code in problems:
                print(f"  Position {pos}: {format_unicode_name(char)}")
            print("\nTo fix this, try:")
            print(f"  export {var_name}=\"your-api-key-here\"")
            print("  (Type the quotes directly in terminal, don't copy-paste)")
    else:
        print("\nAll checked environment variables are ASCII-safe! ✓")

if __name__ == "__main__":
    main()