# patrec
ml workshop codefiles (recognize hand-written numbers) for ourcs 2025 UTA

## folder includes: 
  - intro (intro code)
  - ones_zeros (simple code that has a program recognize if its one or zero)
  - nn (nearest neighbor)
  - nnfast (fastest nearest neighbor code)
  - nnfast2 (2nd fastest nearest neighbor code)
  - nnfast3 (3d fastes nearest neighbor code)
  - knn (this is just a top level k nearest neighbor)
  - cnn (conv neural networks)
  - dnn (dense neural networks)
  - c_d_nn (conv & dense neural networks)

## Notes: 
  1. nnfast is numbered based from most to least fastest
  2. Since most nnfiles contain similar functions, you'll find that some
     code functions haven't been commented or explained (most likely its in another nn- file)
  3. cnn.py and dnn.py are a bit more complicated as they use batch processing and optimize cpu cores
       - this is for people that don't have CUDA (me!). c_d_nn program exec took painfully long!
       - if you have an NVIDIA gpu or are using collab or something, use c_d_nn (much simpler and intuitive
  4. I haven't explained the neural networks codefiles yet as I myself don't understand them that well (will update soon!)
  5. You can manually play with sizes. (training_images = 60000, test_images = 10000, k = {whatever you think of})
  6. Since this is part of a workshop, will add findings soon!

ciao!

