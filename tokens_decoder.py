def dummy_processor(token_gen):
    buffer = ""
    count = 0
    for token in token_gen:
        # Append the token (which may be a string of text) to the buffer
        buffer += token
        count += 1
        if count == 7:
            yield buffer
            buffer = ""
            count = 0
    # Emit any remaining tokens (if fewer than 7)
    if buffer:
        yield buffer