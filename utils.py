import logging


def chunk(stream, chunk_size):
    buf = []

    for item in stream:
        buf.append(item)

        if len(buf) >= chunk_size:
            yield buf
            del buf[:]

    if len(buf) > 0:
        yield buf


def split_column(m, col_sizes):
    offset = 0
    splits = []

    for colsize in col_sizes:
        splits.append(m[:, offset:(offset + colsize)])
        offset += colsize

    assert offset == m.shape[1]
    return splits


def config_logging(fname):
    logging.basicConfig(level=logging.INFO, format='%(message)s')  # re-format to remove prefix 'INFO:root'

    fh = logging.FileHandler(fname)
    fh.setLevel(logging.INFO)
    logging.getLogger("").addHandler(fh)
