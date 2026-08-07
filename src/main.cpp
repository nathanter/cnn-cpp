#include "layers.hpp"
#include "csv_helpers/image_to_csv.h"
#include "csv_helpers/read_csv.h"
#include "csv_helpers/write_to_csv.h"
#include "math_helpers.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>

using namespace std;

namespace
{

    constexpr int kWidth = 28;
    constexpr int kHeight = 28;
    constexpr int kLabels = 10;
    constexpr int kConvNodes = 32;
    constexpr int kEpochs = 17;
    constexpr int kDenseInputSize = ((kWidth - 2) / 2) * ((kHeight - 2) / 2) * kConvNodes;

    const filesystem::path kTrainCsv = "datasets/Mnist-fashion/fashion-mnist_train.csv";
    const filesystem::path kTestCsv = "datasets/Mnist-fashion/fashion-mnist_test.csv";
    const filesystem::path kDefaultModelDir = "models/fashion-mnist";

    void print_usage(const char *program_name)
    {
        cout << "Usage:\n"
             << "  " << program_name << " train [model_dir] [hidden_layer_sizes...]\n"
             << "  " << program_name << " eval [model_dir] [input_path] [hidden_layer_sizes...]\n"
             << "\n"
             << "hidden_layer_sizes are the node counts of the dense layers before\n"
             << "the final " << kLabels << "-label output layer, which is always added.\n";
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

    vector<Layer> build_dense_layers(const vector<int> &hidden_sizes)
    {
        vector<Layer> dense_layers;
        dense_layers.reserve(hidden_sizes.size() + 1);
        int input_size = kDenseInputSize;
        for (const int hidden_size : hidden_sizes)
        {
            dense_layers.emplace_back(hidden_size, input_size);
            input_size = hidden_size;
        }
        dense_layers.emplace_back(kLabels, input_size);
        return dense_layers;
    }

    void initialize_model(Conv_Layer2D &conv_layer, vector<Layer> &dense_layers)
    {
        conv_layer.Rand_filter();
        for (Layer &dense_layer : dense_layers)
        {
            dense_layer.randomize_weights();
            dense_layer.randomize_biases();
        }
    }

    Tensor run_inference(Conv_Layer2D &conv_layer, Pool_Layer2x2 &pool_layer,
                         vector<Layer> &dense_layers, Tensor &image)
    {
        Tensor result = conv_layer.feed_forward(image);
        apply_relu_activation(result);
        result = pool_layer.Pool(result);
        for (int layer = 0; layer < static_cast<int>(dense_layers.size()); layer++)
        {
            result = dense_layers[layer].feed_forward(result);
            if (layer + 1 < static_cast<int>(dense_layers.size()))
            {
                apply_relu_activation(result);
            }
        }
        apply_soft_max(result);
        return result;
    }

    float evaluate_csv_dataset(const vector<vector<float>> &csv_test_data,
                               Conv_Layer2D &conv_layer, Pool_Layer2x2 &pool_layer,
                               vector<Layer> &dense_layers)
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
            Tensor result = run_inference(conv_layer, pool_layer, dense_layers, image);
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
                    const vector<Layer> &dense_layers)
    {
        filesystem::create_directories(model_dir);
        write_tensor_to_csv(conv_layer.data, model_dir / "conv_filters.csv");
        write_tensor_to_csv(conv_layer.biases_data, model_dir / "conv_biases.csv");
        for (int layer = 0; layer < static_cast<int>(dense_layers.size()); layer++)
        {
            write_tensor_to_csv(dense_layers[layer].weights,
                                model_dir / ("dense_weights_" + to_string(layer) + ".csv"));
            write_tensor_to_csv(dense_layers[layer].biases,
                                model_dir / ("dense_biases_" + to_string(layer) + ".csv"));
        }
    }

    void load_model(const filesystem::path &model_dir, Conv_Layer2D &conv_layer,
                    vector<Layer> &dense_layers)
    {
        conv_layer.data =
            read_tensor_from_csv(model_dir / "conv_filters.csv", {kConvNodes, 3, 3});
        conv_layer.biases_data =
            read_tensor_from_csv(model_dir / "conv_biases.csv", {kConvNodes});

        // looks at how many dense layers we expect.
        // naming schema is dense_weights_{position in dense layer list}.csv
        // other direction(saving) does this so output layer should be last one
        for (int layer = 0; layer < static_cast<int>(dense_layers.size()); layer++)
        {
            Layer &dense_layer = dense_layers[layer];
            dense_layer.weights = read_tensor_from_csv(
                model_dir / ("dense_weights_" + to_string(layer) + ".csv"),
                {dense_layer.nodes, dense_layer.inputs});
            dense_layer.biases = read_tensor_from_csv(
                model_dir / ("dense_biases_" + to_string(layer) + ".csv"),
                {dense_layer.nodes});
        }
    }

    void train_command(const filesystem::path &model_dir, const vector<int> &hidden_sizes)
    {
        // for training. Applies relu after each inner layer and softmax at final layer. if only one layer as in previous versions then only softmax is run
        Conv_Layer2D conv_layer(kConvNodes);
        Pool_Layer2x2 pool_layer;
        vector<Layer> dense_layers = build_dense_layers(hidden_sizes);

        initialize_model(conv_layer, dense_layers);


        float learning_rate = 0.01f;

        const auto csv_data = read_csv_rows(kTrainCsv, true);
        const auto csv_test_data = read_csv_rows(kTestCsv, true);

        if (csv_data.empty())
        {
            throw runtime_error("Training dataset is empty");
        }

        Tensor image(kHeight, kWidth);
        vector<int> row_indexes(csv_data.size());


        // intialize this so we can rearrange training order
        for (int index = 0; index < static_cast<int>(row_indexes.size()); index++)
        {
            row_indexes[index] = index;
        }

        std::mt19937 rng(std::random_device{}());
        cout << "Starting training. Epoch " << 0 << "/" << kEpochs << '\n';

        // training loop
        for (int epoch = 0; epoch < kEpochs; epoch++)
        {
            if (epoch %5 == 0 && epoch != 0)
            {
                learning_rate /= 2.0f;
                conv_layer.update_learning_rate(learning_rate);
                for (Layer &dense_layer : dense_layers)
                {
                    dense_layer.update_learning_rate(learning_rate);
                }
            }

            shuffle(row_indexes.begin(), row_indexes.end(), rng);

            float epoch_loss = 0.0f;
            int epoch_correct = 0;



            for (const int row_index : row_indexes)
            {
                int label = 0;
                image = csv_line_to_tensor(csv_data[row_index], label, image);

                Tensor result = conv_layer.feed_forward(image);
                Tensor conv_relu_mask = apply_relu_activation(result);

                result = pool_layer.Pool(result);

                vector<Tensor> dense_relu_masks;
                for (int layer = 0; layer < static_cast<int>(dense_layers.size()); layer++)
                {
                    result = dense_layers[layer].feed_forward(result);
                    if (layer + 1 < static_cast<int>(dense_layers.size())) //keeps last layer for softmax
                    {
                        dense_relu_masks.push_back(apply_relu_activation(result));
                    }
                }

                Tensor loss = soft_max_opertation(result, label);

                // result now holds softmax probabilities, so cross-entropy
                // loss for this sample is -log(probability of correct label)
                epoch_loss += -log(max(result.get_element_flat(label), 1e-7f));
                if (get_max(result) == label)
                {
                    epoch_correct++;
                }

                for (int layer = static_cast<int>(dense_layers.size()) - 1; layer >= 0; layer--)
                {
                    loss = dense_layers[layer].back_prop(loss);
                    if (layer > 0)
                    {
                        loss = apply_relu_back_prop(dense_relu_masks[layer - 1], loss);
                    }
                }
                loss = pool_layer.back_prop(loss);
                loss = apply_relu_back_prop(conv_relu_mask, loss);

                conv_layer.backprop(loss);
            }

            cout << "Finished epoch " << (epoch + 1) << "/" << kEpochs
                 << " avg loss: " << epoch_loss / row_indexes.size()
                 << " train accuracy: "
                 << 100.0f * epoch_correct / row_indexes.size() << "%" << endl;
        }

        save_model(model_dir, conv_layer, dense_layers);
        const float accuracy =
            evaluate_csv_dataset(csv_test_data, conv_layer, pool_layer, dense_layers);

        cout << "Saved model to " << model_dir << '\n';
        cout << "accuracy: " << accuracy << '\n';
    }

    void eval_command(const filesystem::path &model_dir,
                      const filesystem::path &input_path,
                      const vector<int> &hidden_sizes)
    {
        // as with training uses softmax on the last layer
        Conv_Layer2D conv_layer(kConvNodes);
        Pool_Layer2x2 pool_layer;
        vector<Layer> dense_layers = build_dense_layers(hidden_sizes);
        load_model(model_dir, conv_layer, dense_layers);

        if (input_path.extension() == ".csv")
        {
            const auto csv_rows = read_csv_rows(input_path, true);
            const float accuracy =
                evaluate_csv_dataset(csv_rows, conv_layer, pool_layer, dense_layers);
            cout << "accuracy: " << accuracy << '\n';
            return;
        }

        Tensor image(kHeight, kWidth);
        image = image_file_to_tensor(input_path.string(), image);
        Tensor result = run_inference(conv_layer, pool_layer, dense_layers, image);
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

            vector<int> hidden_sizes;
            for (int i = 3; i < argc; i++)
            {
                const int size = stoi(argv[i]);
                if (size <= 0)
                {
                    throw runtime_error("hidden layer sizes must be positive");
                }
                hidden_sizes.push_back(size);
            }

            train_command(model_dir, hidden_sizes);

            return 0;
        }

        if (command_string == "eval")
        {
            const filesystem::path model_dir =
                argc >= 3 ? filesystem::path(argv[2]) : kDefaultModelDir;
            const filesystem::path input_path =
                argc >= 4 ? filesystem::path(argv[3]) : kTestCsv;

            vector<int> hidden_sizes;
            for (int i = 4; i < argc; i++)
            {
                const int size = stoi(argv[i]);
                if (size <= 0)
                {
                    throw runtime_error("hidden layer sizes must be positive");
                }
                hidden_sizes.push_back(size);
            }

            eval_command(model_dir, input_path, hidden_sizes);
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
