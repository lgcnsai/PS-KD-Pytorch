import numpy as np
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description='visualizer for class prototype vectors')
    parser.add_argument('--experiment_dir', type=str, default='expts',
                        help='Directory name where the model ckpts are stored')
    args = parser.parse_args()
    return parser, args


def main():
    parser, args = parse_args()
    batch_size = 256
    num_classes = 100
    embed_dim = 512
    teach_embed_dim = 200

    num_epochs = np.load(os.path.join(args.experiment_dir, "model/embeddings.npy")).shape[0]

    for _ in range(num_epochs):
        embeddings = np.load(os.path.join(args.experiment_dir,"model/embeddings.npy"))[_]  # batch_size by embed_dim
        learnable_parameters = np.load(os.path.join(args.experiment_dir, "model/learnable_parameters.npy"))[_]  # num_classes by teach_embed_dim
        learnable_parameters_similarity = np.load(os.path.join(args.experiment_dir,
                                                               "model/learnable_parameters_similarity.npy"))[_]
        # num_classes by num_classes
        teacher_output_before_learnable = np.load(os.path.join(args.experiment_dir,
                                                               "model/teacher_output_before_learnable.npy"))[_]
        # batch_size by teach_embed_dim
        teacher_logits = np.load(os.path.join(args.experiment_dir, "model/teacher_logits.npy"))[_]  # batch_size by num_classes

        normalized_learnable_parameters = np.linalg.norm(learnable_parameters, ord=2, axis=1)
        print(f"normalized learnable parameters are {num_classes} vectors; "
              f"(Actual shape: {normalized_learnable_parameters.shape}) "
              f"and should have all values equal to 1")  # <- confirmed
        print("normalized learnable parameters:")
        print(normalized_learnable_parameters)
        print("\n\n")

        print(f"similarities is a symmetric matrix with values between -1 and 1, and ones on the diagonal")  # <- confirmed
        print(f"smallest similarity: {np.min(learnable_parameters_similarity)}")
        print(f"biggest similarity: {np.max(learnable_parameters_similarity)}")
        print("similarity matrix")
        print(learnable_parameters_similarity)
        print("diagonal of similarity matrix:")
        print(np.diagonal(learnable_parameters_similarity))
        print("\n\n")

        normalized_teacher_before_learnable = np.linalg.norm(teacher_output_before_learnable, ord=2, axis=1)
        print(f"the normalized teacher embedding before going through the learnable parameters has "
              f"{batch_size} many vectors;"
              f"(Actual shape: {normalized_teacher_before_learnable.shape}) "
              f"and all values are equal to 1")  # <- confirmed
        print("normalized teacher embedding before going through the learnable parameters:")
        print(normalized_teacher_before_learnable)
        print("\n\n")

        print("Teacher logits are values between -1 and 1")  # <- confirmed
        print(f"biggest teacher logit: {np.max(teacher_logits)}")
        print(f"smallest teacher logit: {np.min(teacher_logits)}")

        # teacher_before_learnable @ learnable.T = teacher_logits  # <- somehow not true
        x = teacher_output_before_learnable @ learnable_parameters.T
        z = teacher_logits
        np.testing.assert_allclose(x, z, rtol=1e-9), "The arrays are not equal"
        print('-------------------------------')


if __name__ == '__main__':
    main()