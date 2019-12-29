# Two-Branch Neural Networks

This repo is a pytorch implemention of [Two Branch Networks (Liwei Wang, et al.)](https://github.com/lwwang/Two_branch_network) and has been modified to enable testing different settings as done in [this paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Burns_Language_Features_Matter_Effective_Language_Representations_for_Vision-Language_Tasks_ICCV_2019_paper.pdf) like fine-tuning a word embedding, using different language models.

This code has been tested using python 2.7 and pytorch 1.0.1.

## Datasets:

You can download and unpack the caption data using:

  ```Shell
  ./download.sh
  ```

This doesn't include precomputed visual features or language embeddings.  You can obtain the ResNet-152 visual features we used [here](https://drive.google.com/file/d/1Janoli8suKrk9c4MHR0uIYyCK6o7iNCF/view?usp=sharing).  The code is setup to load word embeddings from a space separated text file.  By default the code will load [MT GrOVLE](http://ai.bu.edu/grovle/) embeddings which it assumes has been placed in the `data` directory.  When tuning the `word_embedding_reg` we found values anywhere between 1.5 and 0 to be optimal depending on the word embedding tested, and tuning this parameter for the word embedding can considerably improve performance.

## Usage:

After setting up the datasets, you can train a model using `main.py`:

  ```Shell
  # Examples:
  python main.py --dataset flickr --language_model attend --name default_attend
  python main.py --dataset coco --language_model attend --name default_attend
  ```

Training using both `avg` and `attend` language models should take less than an hour on a Titan Xp GPU (on Flickr30K, just a few minutes).

By default the test performance of the model that did the best on the validation set will be printed out, but you can retest by using the `--test` flag and load a model using `--resume <model path>`.

When evaluating it's important to note the discrepancy in the splits on the Flickr30K dataset.  At least two (if not more) splits are used to evaluate the dataset on this task.  The difference in performance between different splits can easily account for a 1-2% difference (this is also true on MSCOCO, but there is more stability in splits there).  It isn't clear if one split always gets better performance than other, and without trying many different models on the same splits it can't be known with any certainty.  We use the same splits as provided by [Flickr30K Entities dataset](http://bryanplummer.com/Flickr30kEntities).  Both datasets use the 1K test splits.  

## Example experiments

Below we provide an example of one of our runs training and testing a self-attention language model using the MT GrOVLE embeddings (which is a little better than the results reported [here](http://openaccess.thecvf.com/content_ICCV_2019/papers/Burns_Language_Features_Matter_Effective_Language_Representations_for_Vision-Language_Tasks_ICCV_2019_paper.pdf), and better than the Two Branch Network's original paper):

  ```Shell
  # The three values for each direction correspond to Recall@{1, 5, 10} (6 numbers total), 
  # and mR refers to the mean of the six recall values.

  # For the Flickr30K dataset
  im2sent: 61.7 86.5 93.2 sent2im: 45.6 76.2 85.3 mr: 74.8

  # For the MSCOCO dataset
  im2sent: 68.7 93.5 97.4 sent2im: 54.5 85.6 93.3 mr: 82.2
  ```

These are results using [this fork](https://github.com/BryanPlummer/Two_branch_network) of the official tensorflow implementation, but you should get comparaible performance with this repo.

## References

If you use this repo in your project please cite the following papers on the Two Branch Network:

``` markdown
@inproceedings{wang2016learning,
  title={Learning deep structure-preserving image-text embeddings},
  author={Wang, Liwei and Li, Yin and Lazebnik, Svetlana},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={5005--5013},
  year={2016}
}

@article{wang2019learning,
  title={Learning two-branch neural networks for image-text matching tasks},
  author={Wang, Liwei and Li, Yin and Huang, Jing and Lazebnik, Svetlana},
  journal={TPAMI},
  volume={41},
  number={2},
  pages={394--407},
  year={2019},
  publisher={IEEE}
}
```


In addition, if you use the MT GrOVLE word embeddings, want to compare to performance using ResNet features, or use the self-attention model please also cite:

``` markdown
@InProceedings{burnsLanguage2019,
  title={Language Features Matter: {E}ffective Language Representations for Vision-Language Tasks},
  author={Andrea Burns and Reuben Tan and Kate Saenko and Stan Sclaroff and Bryan A. Plummer},
  booktitle={The IEEE International Conference on Computer Vision (ICCV)},
  year={2019}
}
```

