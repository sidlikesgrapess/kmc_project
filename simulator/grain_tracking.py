_next_grain_id = 0


def reset_grain_counter():
    global _next_grain_id
    _next_grain_id = 0


def new_grain_id():
    global _next_grain_id
    gid = _next_grain_id
    _next_grain_id += 1
    return gid