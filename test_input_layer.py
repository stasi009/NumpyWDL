import numpy as np
from input_layer import SparseInput


# def test_iterate_example_from_sparse_input(example_indices, batch_size):
#     sp_input = SparseInput(example_indices=example_indices,
#                            feature_ids=example_indices,
#                            feature_values=example_indices,
#                            batch_size=batch_size)
#
#     # for example_idx, feat_ids, feat_values in sp_input.iterate_example():
#     #     print("\n**************** {}-th example: ".format(example_idx))
#     #     print("feature ids:    {}".format(feat_ids))
#     #     print("feature values: {}".format(feat_values))
#
#     iterator = sp_input.iterate_example()
#     while True:
#         try:
#             example_idx, feat_ids, feat_values = next(iterator)
#             print("\n**************** {}-th example: ".format(example_idx))
#             print("feature ids:    {}".format(feat_ids))
#             print("feature values: {}".format(feat_values))
#         except StopIteration:
#             break


def test_get_example_in_order_from_sparse(example_indices, batch_size):
    sp_input = SparseInput(example_indices=example_indices,
                           feature_ids=example_indices,
                           feature_values=example_indices,
                           n_total_examples=batch_size)

    for example_idx in range(batch_size):
        feat_ids, feat_vals = sp_input.get_example_in_order(example_idx)
        print("\n**************** {}-th example: ".format(example_idx))
        print("feature ids:    {}".format(feat_ids))
        print("feature values: {}".format(feat_vals))


if __name__ == "__main__":
    test_get_example_in_order_from_sparse(example_indices=[1, 1, 1, 3, 4, 6],batch_size=10)

    # test_get_example_in_order_from_sparse(example_indices=[1, 1, 1, 3, 4, 6],batch_size=3)

    # test_get_example_in_order_from_sparse(example_indices=[0, 1, 1, 1, 3, 4, 7],batch_size=9)

    # test_get_example_in_order_from_sparse(example_indices=[], batch_size=9)
