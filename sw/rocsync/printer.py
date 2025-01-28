from tqdm import tqdm


def print(string):
    tqdm.write(str(string))


def errprint(string):
    print(f"\033[91m{string}\033[0m")


def warnprint(string):
    print(f"\033[93m{string}\033[0m")


def succprint(string):
    print(f"\033[92m{string}\033[0m")


def boldprint(string):
    print(f"\033[1m{string}\033[0m")


def printresult(name, value, is_valid):
    string = f"{name+':':<40} {value:>30}"
    if is_valid:
        succprint(string)
    else:
        errprint(string)
