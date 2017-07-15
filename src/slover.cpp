#include <isotream>
#include "../include/netparam.hpp"
#include "include/slover.hpp"
#include "include/net.hpp"
using namespace std;
namespace lego_net{
void Slover::slove(NetParam& param, vector<shared_ptr<Blob>> X , vector<shared_ptr<Blob>> Y) {
    // to be delete
    X_train_ = X[0];
    Y_train_ = Y[0];
    X_val_ = X[1];
    Y_val_ = Y[1]
    int N = X_train_->get_N();
    int iter_per_epochs;
    if (param.use_batch) {
        iter_per_epochs = N / param.batch_size;
    }
    else {
        iter_per_epochs = N;
    }
    int num_iters = iter_per_epochs * param.num_epochs;
    int epoch = 0;

    // iteration
    for (int iter = 0; iter < num_iters; ++iter) {
        // batch
        shared_ptr<Blob> X_batch;
        shared_ptr<Blob> Y_batch;
        if (param.use_batch) {
            // deep copy
            X_batch.reset(new Blob(X_train_->subBlob((iter * param.batch_size) % N,
                                                        ((iter+1) * param.batch_size) % N)));
            Y_batch.reset(new Blob(Y_train_->subBlob((iter * param.batch_size) % N,
                                                        ((iter+1) * param.batch_size) % N)));
        }
        else {
            shared_ptr<Blob> X_batch = X_train_;
            shared_ptr<Blob> Y_batch = Y_train_;
        }

        // train
        trainNet(X_batch, Y_batch, param);

        // update
        for (int i = 0; i < (int)layers_.size(); ++i) {
            std::string lname = layers_[i];
            if (!data_[lname][1] || !data_[lname][2]) {
                continue;
            }
            for (int j = 1; j <= 2; ++j) {
                assert(param.update == "momentum" ||
                       param.update == "rmsprop" ||
                       param.update == "adagrad" ||
                       param.update == "sgd");
                shared_ptr<Blob> dx(new Blob(data_[lname][j]->size()));
                if (param.update == "sgd") {
                    *dx = -param.lr * (*grads_[lname][j]);
                }
                if (param.update == "momentum") {
                    if (!step_cache_[lname][j]) {
                        step_cache_[lname][j].reset(new Blob(data_[lname][j]->size(), TZEROS));
                    }
                    Blob ll = param.momentum * (*step_cache_[lname][j]);
                    Blob rr = param.lr * (*grads_[lname][j]);
                    *dx = ll - rr;
                    step_cache_[lname][j] = dx;
                }
                if (param.update == "rmsprop") {
                    // change it self
                    double decay_rate = 0.99;
                    if (!step_cache_[lname][j]) {
                        step_cache_[lname][j].reset(new Blob(data_[lname][j]->size(), TZEROS));
                    }
                    Blob r1 = decay_rate * (*step_cache_[lname][j]);
                    Blob r2 = (1 - decay_rate) * (*grads_[lname][j]);
                    Blob r3 = r2 * (*grads_[lname][j]);
                    *step_cache_[lname][j] = r1 + r3;
                    Blob d1 = (*step_cache_[lname][j]) + 1e-8;
                    Blob u1 = param.lr * (*grads_[lname][j]);
                    Blob d2 = lego_net::sqrt(d1);
                    Blob r4 = u1 / d2;
                    *dx = 0 - r4;
                }
                if (param.update == "adagrad") {
                    if (!step_cache_[lname][j]) {
                        step_cache_[lname][j].reset(new Blob(data_[lname][j]->size(), TZEROS));
                    }
                    *step_cache_[lname][j] = (*grads_[lname][j]) * (*grads_[lname][j]);
                    Blob d1 = (*step_cache_[lname][j]) + 1e-8;
                    Blob u1 = param.lr * (*grads_[lname][j]);
                    Blob d2 = lego_net::sqrt(d1);
                    Blob r4 = u1 / d2;
                    *dx = 0 - r4;
                }
                *data_[lname][j] = (*data_[lname][j]) + (*dx);
            }
        }

        // evaluate
        bool first_it = (iter == 0);
        bool epoch_end = (iter + 1) % iter_per_epochs == 0;
        bool acc_check = (param.acc_frequence && (iter+1) % param.acc_frequence == 0);
        if (first_it || epoch_end || acc_check) {
            // update learning rate[TODO]
            if ((iter > 0 && epoch_end) || param.acc_update_lr) {
                param.lr *= param.lr_decay;
                if (epoch_end) {
                    epoch++;
                }
            }

            // evaluate train set accuracy
            shared_ptr<Blob> X_train_subset;
            shared_ptr<Blob> Y_train_subset;
            if (N > 1000) {
                X_train_subset.reset(new Blob(X_train_->subBlob(0, 100)));
                Y_train_subset.reset(new Blob(Y_train_->subBlob(0, 100)));
            }
            else {
                X_train_subset = X_train_;
                Y_train_subset = Y_train_;
            }
            trainNet(X_train_subset, Y_train_subset, param, "forward");
            double train_acc = prob(*data_[layers_.back()][1], *data_[layers_.back()][0]);
            train_acc_history_.push_back(train_acc);

            // evaluate val set accuracy[TODO: change train to val]
            trainNet(X_val_, Y_val_, param, "forward");
            double val_acc = prob(*data_[layers_.back()][1], *data_[layers_.back()][0]);
            val_acc_history_.push_back(val_acc);

            // print
            printf("iter: %d  loss: %f  train_acc: %0.2f%%    val_acc: %0.2f%%    lr: %0.6f\n",
                    iter, loss_, train_acc*100, val_acc*100, param.lr);

            // save best model[TODO]
            //if (val_acc_history_.size() > 1 && val_acc < val_acc_history_[val_acc_history_.size()-2]) {
            //    for (auto i : layers_) {
            //        if (!data_[i][1] || !data_[i][2]) {
            //            continue;
            //        }
            //        best_model_[i][1] = data_[i][1];
            //        best_model_[i][2] = data_[i][2];
            //    }
            //}
        }
    }

    return;
}
}//end lego_net