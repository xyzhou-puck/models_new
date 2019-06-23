#encoding=utf8

import os
import sys
import numpy as np
import argparse
import paddle
import paddle.fluid as fluid

from arg_config import ArgConfig, print_arguments
from cnn_mnist_net import create_net


def init_from_checkpoint(args, exe, program):

    assert isinstance(args.init_from_checkpoint, str)

    if not os.path.exists(args.init_from_checkpoint):
        raise Warning("the checkpoint path %s does not exist." %
                      args.init_from_checkpoint)
        return False

    fluid.io.load_persistables(
        executor=exe,
        dirname=args.init_from_checkpoint,
        main_program=program,
        filename="checkpoint.pdckpt")

    print("finish init model from checkpoint at %s" %
          (args.init_from_checkpoint))

    return True


def save_checkpoint(args, exe, program, dirname):

    assert isinstance(args.save_model_path, str)

    checkpoint_dir = os.path.join(args.save_model_path, args.save_checkpoint)

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    fluid.io.save_persistables(
        exe,
        os.path.join(checkpoint_dir, dirname),
        main_program=program,
        filename="checkpoint.pdckpt")

    print("save checkpoint at %s" % (os.path.join(checkpoint_dir, dirname)))

    return True


def save_param(args, exe, program, dirname):

    assert isinstance(args.save_model_path, str)

    param_dir = os.path.join(args.save_model_path, args.save_param)

    if not os.path.exists(param_dir):
        os.mkdir(param_dir)

    fluid.io.save_params(
        exe,
        os.path.join(param_dir, dirname),
        main_program=program,
        filename="params.pdparams")

    print("save parameters at %s" % (os.path.join(param_dir, dirname)))

    return True


def do_evaluate_in_training(reader, executor, program, fetch_list):

    outputs = []

    reader.start()

    while True:
        try:
            output = executor.run(program, fetch_list=fetch_list)
            outputs.append(output[0])
        except:
            reader.reset()
            break

    outputs = np.concatenate(outputs).astype("int32")

    testset = paddle.dataset.mnist.test()

    labels = []
    for feature, label in testset():
        labels.append(label)

    labels = np.array(labels).astype("int32")
    acc = (outputs == labels).mean()

    return acc


def do_train(args):

    train_prog = fluid.default_main_program()
    startup_prog = fluid.default_startup_program()

    with fluid.program_guard(train_prog, startup_prog):
        train_prog.random_seed = args.random_seed
        startup_prog.random_seed = args.random_seed

        with fluid.unique_name.guard():

            # define reader

            image = fluid.layers.data(
                name='image', shape=[1, 28, 28], dtype='float32')

            label = fluid.layers.data(name='label', shape=[1], dtype='int64')

            reader = fluid.io.PyReader(
                feed_list=[image, label], capacity=4, iterable=False)

            # define the network

            loss = create_net(
                is_training=True, model_input=[image, label], args=args)

            # define optimizer for learning

            optimizer = fluid.optimizer.Adam(learning_rate=args.learning_rate)
            optimizer.minimize(loss)

    if args.do_eval_in_training:
        # if you want to perform testing in training, you need to declare an extra test-program
        test_prog = fluid.Program()

        with fluid.program_guard(test_prog, startup_prog):
            test_prog.random_seed = args.random_seed

            with fluid.unique_name.guard():

                test_image = fluid.layers.data(
                    name='test_image', shape=[1, 28, 28], dtype='float32')

                test_label = fluid.layers.data(
                    name='test_label', shape=[1], dtype='int64')

                test_reader = fluid.io.PyReader(
                    feed_list=[test_image, test_label],
                    capacity=4,
                    iterable=False)

                predict = create_net(
                    is_training=False, model_input=test_image, args=args)

    # prepare training

    ## declare data generator
    generator = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.mnist.train(), buf_size=500),
        batch_size=args.batch_size)

    reader.decorate_sample_list_generator(generator)

    if args.use_cuda:
        place = fluid.CUDAPlace(0)
    else:
        place = fluid.CPUPlace()

    exe = fluid.Executor(place)
    exe.run(startup_prog)

    if args.init_from_checkpoint:
        init_from_checkpoint(args, exe, train_prog)

    compiled_train_prog = fluid.CompiledProgram(train_prog).with_data_parallel(
        loss_name=loss.name)

    if args.do_eval_in_training:
        compiled_test_prog = fluid.CompiledProgram(test_prog)

        test_generator = paddle.batch(
            paddle.dataset.mnist.test(), batch_size=args.batch_size)
        test_reader.decorate_sample_list_generator(test_generator)

    # start training

    step = 0
    for epoch_step in range(args.epoch_num):
        reader.start()
        while True:
            try:

                # this is for minimizing the fetching op, saving the training speed.
                if step % args.print_step == 0:
                    fetch_list = [loss.name]
                else:
                    fetch_list = []

                output = exe.run(compiled_train_prog, fetch_list=fetch_list)

                if step % args.print_step == 0:
                    print("step: %d, loss: %.4f" % (step, np.sum(output[0])))

                if step % args.save_step == 0 and step != 0:

                    if args.save_checkpoint:
                        save_checkpoint(args, exe, train_prog,
                                        "step_" + str(step))

                    if args.save_param:
                        save_param(args, exe, train_prog, "step_" + str(step))

                if args.do_eval_in_training:
                    if step != 0 and step % args.eval_step == 0:
                        acc = do_evaluate_in_training(
                            reader=test_reader,
                            executor=exe,
                            program=test_prog,
                            fetch_list=[predict.name])
                        print("evaluation acc for step %d is %.4f" %
                              (step, acc))

                step += 1

            except fluid.core.EOFException:
                reader.reset()
                break

    if args.save_checkpoint:
        save_checkpoint(args, exe, train_prog, "step_final")

    if args.save_param:
        save_param(args, exe, train_prog, "step_final")


if __name__ == "__main__":
    args = ArgConfig()
    args = args.build_conf()
    print_arguments(args)

    do_train(args)
