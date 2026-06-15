#include "layers.hpp"
#include "csv_helpers/image_to_csv.h"
#include "csv_helpers/read_csv.h"
#include "csv_helpers/write_to_csv.h"
#include "math_helpers.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>

using namespace std;

namespace
{

    constexpr int kWidth = 28;
    constexpr int kHeight = 28;
    constexpr int kLabels = 10;
    constexpr int kConvNodes = 10;
    constexpr int kEpochs = 3;

    const filesystem::path kTrainCsv = "datasets/Mnist-fashion/fashion-mnist_train.csv";
    const filesystem::path kTestCsv = "datasets/Mnist-fashion/fashion-mnist_test.csv";
    const filesystem::path kDefaultModelDir = "models/fashion-mnist";

    void print_usage(const char *program_name)
    {
        cout << "Usage:\n"
             << "  " << program_name << " train [model_dir]\n"
             << "  " << program_name << " eval [model_dir] [input_path]\n";
    }

    void validate_csv_row(const vector<float> &csv_line)
    {
        if (csv_line.size() != static_cast<size_t>(kWidth * kHeight + 1))
        {
            throw runtime_error("CSV row does not match expected Mnist-fashion shape");
        }
    }

    Tensor csv_line_to_tensor(const vector<float> &csv_line, int &label, Tensor &output)
    {

        // takes one vector of float values to a tensor normalized to 0-1 by dividing
        // by 255.0f

        validate_csv_row(csv_line);
        label = static_cast<int>(csv_line[0]);

        vector<float> &o_d = output.data;
        for (int pix = 1; pix < static_cast<int>(csv_line.size()); pix++)
        {
            o_d[pix - 1] = csv_line[pix] / 255.0f;
        }

        return output;
    }

    void initialize_model(Conv_Layer2D &conv_layer, Layer &dense_layer)
    {
        conv_layer.Rand_filter();
        dense_layer.randomize_weights();
        dense_layer.randomize_biases();
    }

    Tensor run_inference(Conv_Layer2D &conv_layer, Pool_Layer2x2 &pool_layer,
                         Layer &dense_layer, Tensor &image)
    {
        Tensor result = conv_layer.feed_forward(image);
        apply_relu_activation(result);
        result = pool_layer.Pool(result);
        result = dense_layer.feed_forward(result);
        apply_soft_max(result);
        return result;
    }

    float evaluate_csv_dataset(const vector<vector<float>> &csv_test_data,
                               Conv_Layer2D &conv_layer, Pool_Layer2x2 &pool_layer,
                               Layer &dense_layer)
    {
        if (csv_test_data.empty())
        {
            throw runtime_error("Evaluation dataset is empty");
        }

        Tensor image(kHeight, kWidth);
        int correct = 0;
        int testing_label = 0;
        int total_entries = csv_test_data.size();
        int expected_percent = 0.0f;

        for (int entry = 0; entry < static_cast<int>(csv_test_data.size()); entry++)
        {
            image = csv_line_to_tensor(csv_test_data[entry], testing_label, image);
            Tensor result = run_inference(conv_layer, pool_layer, dense_layer, image);
            if (testing_label == get_max(result))
            {
                correct++;
            }
            // progression print
            if ((100.0f * static_cast<float>(correct) /
                 static_cast<float>(total_entries)) > expected_percent)
            {
                cout << "Currently Finished " << expected_percent << "percent of test entries" << endl;
                expected_percent += 25.0f;
            }
        }
        cout << "Currently Finished 100 percent of test entries" << endl;
        return 100.0f * static_cast<float>(correct) /
               static_cast<float>(total_entries);
    }

    void save_model(const filesystem::path &model_dir, const Conv_Layer2D &conv_layer,
                    const Layer &dense_layer)
    {
        filesystem::create_directories(model_dir);
        write_tensor_to_csv(conv_layer.data, model_dir / "conv_filters.csv");
        write_tensor_to_csv(conv_layer.biases_data, model_dir / "conv_biases.csv");
        write_tensor_to_csv(dense_layer.weights, model_dir / "dense_weights.csv");
        write_tensor_to_csv(dense_layer.biases, model_dir / "dense_biases.csv");
    }

    void load_model(const filesystem::path &model_dir, Conv_Layer2D &conv_layer,
                    Layer &dense_layer)
    {
        conv_layer.data =
            read_tensor_from_csv(model_dir / "conv_filters.csv", {kConvNodes, 3, 3});
        conv_layer.biases_data =
            read_tensor_from_csv(model_dir / "conv_biases.csv", {kConvNodes});
        dense_layer.weights = read_tensor_from_csv(
            model_dir / "dense_weights.csv",
            {kLabels, ((kWidth - 2) / 2) * ((kHeight - 2) / 2) * kConvNodes});
        dense_layer.biases =
            read_tensor_from_csv(model_dir / "dense_biases.csv", {kLabels});
    }

    void train_command(const filesystem::path &model_dir)
    {
        Conv_Layer2D conv_layer(kConvNodes);
        Pool_Layer2x2 pool_layer;
        Layer dense_layer(kLabels, ((kWidth - 2) / 2) * ((kHeight - 2) / 2) * kConvNodes);
        initialize_model(conv_layer, dense_layer);

        const auto csv_data = read_csv_rows(kTrainCsv, true);
        const auto csv_test_data = read_csv_rows(kTestCsv, true);
        if (csv_data.empty())
        {
            throw runtime_error("Training dataset is empty");
        }

        Tensor image(kHeight, kWidth);
        vector<int> row_indexes(csv_data.size());
        for (int index = 0; index < static_cast<int>(row_indexes.size()); index++)
        {
            row_indexes[index] = index;
        }

        std::mt19937 rng(std::random_device{}());
        cout << "Starting training. Epoch " << 0 << "/" << kEpochs << '\n';
        for (int epoch = 0; epoch < kEpochs; epoch++)
        {
            shuffle(row_indexes.begin(), row_indexes.end(), rng);

            for (const int row_index : row_indexes)
            {
                int label = 0;
                image = csv_line_to_tensor(csv_data[row_index], label, image);

                Tensor result = conv_layer.feed_forward(image);
                Tensor relu_back_helper = apply_relu_activation(result);
                result = pool_layer.Pool(result);
                result = dense_layer.feed_forward(result);

                Tensor loss = soft_max_opertation(result, label);
                loss = dense_layer.back_prop(loss);
                loss = pool_layer.back_prop(loss);
                loss = apply_relu_back_prop(relu_back_helper, loss);
                conv_layer.backprop(loss);
            }

            cout << "Finished epoch " << (epoch + 1) << "/" << kEpochs << '\n';
            // progression print
        }

        save_model(model_dir, conv_layer, dense_layer);
        const float accuracy =
            evaluate_csv_dataset(csv_test_data, conv_layer, pool_layer, dense_layer);

        cout << "Saved model to " << model_dir << '\n';
        cout << "accuracy: " << accuracy << '\n';
    }

    void eval_command(const filesystem::path &model_dir,
                      const filesystem::path &input_path)
    {

        Conv_Layer2D conv_layer(kConvNodes);
        Pool_Layer2x2 pool_layer;
        Layer dense_layer(kLabels, ((kWidth - 2) / 2) * ((kHeight - 2) / 2) * kConvNodes);
        load_model(model_dir, conv_layer, dense_layer);

        if (input_path.extension() == ".csv")
        {
            const auto csv_rows = read_csv_rows(input_path, true);
            const float accuracy =
                evaluate_csv_dataset(csv_rows, conv_layer, pool_layer, dense_layer);
            cout << "accuracy: " << accuracy << '\n';
            return;
        }

        Tensor image(kHeight, kWidth);
        image = image_file_to_tensor(input_path.string(), image);
        Tensor result = run_inference(conv_layer, pool_layer, dense_layer, image);
        cout << "prediction: " << get_max(result) << '\n';
    }

} // namespace

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        print_usage(argv[0]);
        return 1;
    }

    try
    {
        const string command_string = argv[1];

        if (command_string == "train")
        {
            cout << "training started" << endl;
            const filesystem::path model_dir =
                argc >= 3 ? filesystem::path(argv[2]) : kDefaultModelDir;
            train_command(model_dir);

            return 0;
        }

        if (command_string == "eval")
        {
            const filesystem::path model_dir =
                argc >= 3 ? filesystem::path(argv[2]) : kDefaultModelDir;
            const filesystem::path input_path =
                argc >= 4 ? filesystem::path(argv[3]) : kTestCsv;

            eval_command(model_dir, input_path);
            return 0;
        }

        print_usage(argv[0]);
        cerr << "Invalid command: " << command_string << '\n';
        return 1;
    }
    catch (const exception &error)
    {
        cerr << error.what() << '\n';
        return 1;
    }
}
