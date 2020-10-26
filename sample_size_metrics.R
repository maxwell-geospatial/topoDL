#Read in libraries and connect to Python
library(reticulate)
use_python(python="C:/Users/amaxwel6/anaconda3/envs/envdl/python.exe", required=TRUE) #A custom environment
Sys.setenv(TZ = "America/New_York") #Had to set time zone on VM
library(unet)
library(keras)
library(tfdatasets)
library(tidyverse)
library(rsample)
library(stringr)
library(magick)
library(dplyr)
#List all folders in Kentucky image chip directory, one folder for each SCANID
topo_folders <- as.data.frame(list.dirs("E:/topo_proj/topo_check/processing/ky_chips", full.names= TRUE, recursive=FALSE))
names(topo_folders) <- "folder"

#Loop to extract components of file paths to columns
ky_tab <- data.frame()
for(i in topo_folders) {
  #Load in raster
  #Extract Id
  ky_All <- str_split(i, "_", simplify=TRUE)
  ky_tab <- rbind(ky_tab,ky_All)
}

#Change columns names
names(ky_tab) <- c("Path1", "Path2", "Path3", "Path4", "name", "scanid", "year", "scale", "geo")

#Reconstruct path in new column
ky_tab$path <- paste0(ky_tab$Path1, "_", ky_tab$Path2, "_", ky_tab$Path3, "_", ky_tab$Path4, "_", ky_tab$name, "_", ky_tab$scanid, "_", ky_tab$year, "_", ky_tab$scale, "_", ky_tab$geo)
#Path to background only data
ky_tab$path_b <- paste0("E:/topo_proj/topo_check/processing/ky_chips_background/", "KY_", ky_tab$name, "_", ky_tab$scanid, "_", ky_tab$year, "_", ky_tab$scale, "_", ky_tab$geo)

#Randomly shuffle rows
set.seed(42)
rows <- sample(nrow(ky_tab))
ky_tab2 <- ky_tab[rows,]

#List all topo names
quad_names <- as.data.frame(levels(as.factor(ky_tab2$name)))
names(quad_names) <- "name"

#Split topos into training, testing, and validation sets
set.seed(42)
topos_train <- quad_names %>% sample_n(70)
topos_remaining <- setdiff(quad_names, topos_train)
set.seed(43)
topos_val <- topos_remaining %>% sample_frac(.5)
topos_test <- setdiff(topos_remaining, topos_val)
topos_train$select <- 1
topos_val$select <- 2
topos_test$select <- 3
topos_combined <- rbind(topos_train, topos_val, topos_test)

#Join sampling results back to folder list
ky_tab3 <- left_join(ky_tab2, topos_combined, by="name")

#Separate into training and validation split
topo_folders_train <- ky_tab3 %>% filter(select==1)
topo_folders_val <- ky_tab3 %>% filter(select==2)
topo_folders_test <- ky_tab3 %>% filter(select==3)

#List all folders in Ohio and Virginia image chip directory, one folder for each SCANID
oh_folders <- as.data.frame(list.dirs("E:/topo_proj/topo_check/processing/oh_chips", full.names= TRUE, recursive=FALSE))
names(oh_folders) <- "folder"
va_folders <- as.data.frame(list.dirs("E:/topo_proj/topo_check/processing/va_chips", full.names= TRUE, recursive=FALSE))
names(oh_folders) <- "folder"

#Loop to extract components of file paths to columns
oh_tab <- data.frame()
for(i in oh_folders) {
  #Load in raster
  #Extract Id
  oh_All <- str_split(i, "_", simplify=TRUE)
  oh_tab <- rbind(oh_tab,oh_All)
}

#Change columns names
names(oh_tab) <- c("Path1", "Path2", "Path3", "Path4", "name", "scanid", "year", "scale", "geo")
#Paths to data and background only data
oh_tab$path <- paste0(oh_tab$Path1, "_", oh_tab$Path2, "_", oh_tab$Path3, "_", oh_tab$Path4, "_", oh_tab$name, "_", oh_tab$scanid, "_", oh_tab$year, "_", oh_tab$scale, "_",oh_tab$geo)
oh_tab$path_b <- paste0("E:/topo_proj/topo_check/processing/oh_chips_background/", "OH_", oh_tab$name, "_", oh_tab$scanid, "_", oh_tab$year, "_", oh_tab$scale, "_", oh_tab$geo)

#Loop to extract components of file paths to columns
va_tab <- data.frame()
for(i in va_folders) {
  #Load in raster
  #Extract Id
  va_All <- str_split(i, "_", simplify=TRUE)
  va_tab <- rbind(va_tab,va_All)
}

#Change columns names
names(va_tab) <- c("Path1", "Path2", "Path3", "Path4", "name", "scanid", "year", "scale", "geo")
#Paths to data and background only data
va_tab$path <- paste0(va_tab$Path1, "_", va_tab$Path2, "_", va_tab$Path3, "_", va_tab$Path4, "_", va_tab$name, "_", va_tab$scanid, "_", va_tab$year, "_", va_tab$scale, "_", va_tab$geo)
va_tab$path_b <- paste0("E:/topo_proj/topo_check/processing/va_chips_background/", "VA_", va_tab$name, "_", va_tab$scanid, "_", va_tab$year, "_", va_tab$scale, "_", va_tab$geo)

oh_folders2 <- oh_tab
va_folders2 <- va_tab

#Plot example images and masks with magick
trainPlot <- tibble(
  img=list.files(paste0(as.character(topo_folders_train$path), "/images"), full.names=TRUE, pattern="\\.png$"),
  mask=list.files(paste0(as.character(topo_folders_train$path), "/labels2"), full.names=TRUE, pattern="\\.png$")
)%>% 
  sample_n(2) %>% 
  map(. %>% magick::image_read() %>% magick::image_normalize() %>% magick::image_resize("128x128")) 
out <- magick::image_append(c(
  magick::image_append(trainPlot$img, stack = TRUE), 
  magick::image_append(trainPlot$mask, stack = TRUE)
)
)
plot(out)

#Build training, testing, and validation chip lists
train <- tibble(
  img=list.files(paste0(as.character(topo_folders_train$path), "/images"), full.names=TRUE, pattern="\\.png$"),
  mask=list.files(paste0(as.character(topo_folders_train$path), "/labels2"), full.names=TRUE, pattern="\\.png$")
)

val <- tibble(
  img=list.files(paste0(as.character(topo_folders_val$path), "/images"), full.names=TRUE, pattern="\\.png$"),
  mask=list.files(paste0(as.character(topo_folders_val$path), "/labels2"), full.names=TRUE, pattern="\\.png$")
)

test <- tibble(
  img=list.files(paste0(as.character(topo_folders_test$path), "/images"), full.names=TRUE, pattern="\\.png$"),
  mask=list.files(paste0(as.character(topo_folders_test$path), "/labels2"), full.names=TRUE, pattern="\\.png$")
)

oh <- tibble(
  img=list.files(paste0(as.character(oh_folders2$path), "/images"), full.names=TRUE, pattern="\\.png$"),
  mask=list.files(paste0(as.character(oh_folders2$path), "/labels2"), full.names=TRUE, pattern="\\.png$")
)

va <- tibble(
  img=list.files(paste0(as.character(va_folders2$path), "/images"), full.names=TRUE, pattern="\\.png$"),
  mask=list.files(paste0(as.character(va_folders2$path), "/labels2"), full.names=TRUE, pattern="\\.png$")
)

#List all background chips
train_b <- tibble(
  img=list.files(paste0(as.character(topo_folders_train$path_b), "/images2"), full.names=TRUE, pattern="\\.png$"),
  mask=list.files(paste0(as.character(topo_folders_train$path_b), "/labels2"), full.names=TRUE, pattern="\\.png$")
)

val_b <- tibble(
  img=list.files(paste0(as.character(topo_folders_val$path_b), "/images2"), full.names=TRUE, pattern="\\.png$"),
  mask=list.files(paste0(as.character(topo_folders_val$path_b), "/labels2"), full.names=TRUE, pattern="\\.png$")
)

test_b <- tibble(
  img=list.files(paste0(as.character(topo_folders_test$path_b), "/images2"), full.names=TRUE, pattern="\\.png$"),
  mask=list.files(paste0(as.character(topo_folders_test$path_b), "/labels2"), full.names=TRUE, pattern="\\.png$")
)

oh_b <- tibble(
  img=list.files(paste0(as.character(oh_folders2$path_b), "/images2"), full.names=TRUE, pattern="\\.png$"),
  mask=list.files(paste0(as.character(oh_folders2$path_b), "/labels2"), full.names=TRUE, pattern="\\.png$")
)

va_b <- tibble(
  img=list.files(paste0(as.character(va_folders2$path_b), "/images2"), full.names=TRUE, pattern="\\.png$"),
  mask=list.files(paste0(as.character(va_folders2$path_b), "/labels2"), full.names=TRUE, pattern="\\.png$")
)

#Extract directory path to new column
train$quad <-dirname(train$img)
val$quad <- dirname(val$img)
test$quad <-dirname(test$img)
oh$quad <- dirname(oh$img)
va$quad <- dirname(va$img)

#Extract directory path to new column
train_b$quad <-dirname(train_b$img)
val_b$quad <- dirname(val_b$img)
test_b$quad <-dirname(test_b$img)
oh_b$quad <- dirname(oh_b$img)
va_b$quad <- dirname(va_b$img)

#Randomly select 150 background-only chips from each unique map
set.seed(42)
train_b2 <- train_b %>% group_by(quad) %>% sample_n(150, replace=FALSE)
set.seed(42)
val_b2 <- val_b %>% group_by(quad) %>% sample_n(150, replace=FALSE)
set.seed(42)
test_b2 <- test_b %>% group_by(quad) %>% sample_n(150, replace=FALSE)
set.seed(42)
oh_b2 <- oh_b %>% group_by(quad) %>% sample_n(150, replace=FALSE)
set.seed(42)
va_b2 <- va_b %>% group_by(quad) %>% sample_n(150, replace=FALSE)

#Combine chips and 150 randomly selected background chips to a single object
train_2 <- rbind(train, train_b2)[,1:2]
val_2 <- rbind(val, val_b2)[,1:2]
test_2 <- rbind(test, test_b2)[,1:2]
oh_2 <- rbind(oh, oh_b2)[,1:2]
va_2 <- rbind(va, va_b2)[,1:2]

#Reshuffle data to reduce autocorrelation
set.seed(42)
rows <- sample(nrow(train_2))
train2 <- train_2[rows,]

set.seed(42)
rows <- sample(nrow(val_2))
val2 <- val_2[rows,]

set.seed(42)
rows <- sample(nrow(test_2))
test2 <- test_2[rows,]

set.seed(42)
rows <- sample(nrow(oh_2))
oh2 <- oh_2[rows,]

set.seed(42)
rows <- sample(nrow(va_2))
va2 <- va_2[rows,]

#Define some modeling settings
epochs <- 100L
batch_size <- 32L
chip_size <- 128
threshold <- 0.5

#Create functions to randomly flip the images and masks, common seed is used so that images and masks align
imgAug <- function(img, rndint) {
  img %>%
    tf$image$random_flip_up_down(seed=rndint) %>%
    tf$image$random_flip_left_right(seed=rndint)
}

maskAug <- function(mask, rndint) {
  mask %>%
    tf$image$random_flip_up_down(seed=rndint) %>%
    tf$image$random_flip_left_right(seed=rndint)
}

#Create function to change brightness, contrast, saturation, and hue randomly
random_bsh <- function(img) {
  img %>% 
    tf$image$random_brightness(max_delta = 0.1) %>% 
    tf$image$random_contrast(lower = 0.95, upper = 1.05) %>% 
    tf$image$random_saturation(lower = 0.95, upper = 1.05) %>% 
    tf$image$random_hue(max_delta = 0.1) %>%
    # make sure we still are between 0 and 1
    tf$clip_by_value(0, 1) 
}

#Define data generation function
create_dataset <- function(data, train, batch_size =32L) {

    dataset <- data %>% 
    	tensor_slices_dataset() %>% 
    	dataset_map(~.x %>% list_modify(
      		img = tf$image$decode_png(tf$io$read_file(.x$img)),
      		mask = tf$image$rgb_to_grayscale(tf$image$decode_png(tf$io$read_file(.x$mask), channels=3))
    	)) %>%
   	 dataset_map(~.x %>% list_modify(
      		img = tf$image$convert_image_dtype(.x$img, dtype = tf$float32),
      		mask = tf$image$convert_image_dtype(.x$mask, dtype = tf$float32)
    	)) %>% 
    	dataset_map(~.x %>% list_modify(
     		img = tf$image$resize(.x$img, size = shape(128, 128)),
      		mask = tf$image$resize(.x$mask, size = shape(128, 128), method="nearest")
   	))
  
   if (train) {
    rndint = sample(1:100, 1)
    dataset <- dataset %>% 
      dataset_map(~.x %>% list_modify(
        img = imgAug(.x$img, rndint=rndint),
	mask = maskAug(.x$mask, rndint=rndint)
      )) 
  }

# data augmentation performed on training set only
  if (train) {
    dataset <- dataset %>% 
      dataset_map(~.x %>% list_modify(
        img = random_bsh(.x$img)
      )) 
  }
  
# shuffling on training set only
  if (train) {
    dataset <- dataset %>% 
    dataset_shuffle(buffer_size = 30392) 
  }

  dataset <- dataset %>% 
    dataset_batch(batch_size)
 
  dataset %>% 
    dataset_map(unname) # Keras needs an unnamed output.
}

#Create training, validation, and testing data sets
train_set <- create_dataset(train2, train=FALSE)
val_set <- create_dataset(val2, train=FALSE)
test_set <- create_dataset(test2, train=FALSE)
oh_set <- create_dataset(oh2, train=FALSE)
va_set <- create_dataset(va2, train=FALSE)

#Plot an example to make sure ranges are correct and augmentations are working correctly
example <- test_set %>% as_iterator() %>% iter_next()
example[[1]] %>% as.array() %>% max() 
example[[2]] %>% as.matrix() %>% max()
example[[1]] %>% as.array() %>% min() 
example[[2]] %>% as.matrix() %>% min()
example[[1]][1,,,] %>% as.array() %>% as.raster() %>% plot()
example_mask <- example[[2]][1,,,] %>% as.matrix()
example_mask_fac <- as.factor(example_mask)
levels(example_mask_fac)
example_mask <- array(c(example_mask, example_mask, example_mask), dim=c(128,128, 3))
example_mask %>% as.raster() %>% plot()

# U-net 128 -----------------------------------------------------
#Modified from: https://keras.rstudio.com/articles/examples/unet.html

get_unet_128 <- function(input_shape = c(128, 128, 3),
                         num_classes = 1) {
  
  inputs <- layer_input(shape = input_shape)
  # 128
  
  down1 <- inputs %>%
    layer_conv_2d(filters = 64, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation_leaky_relu()%>% #Changed activation to Leaky RELU
    layer_conv_2d(filters = 64, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation_leaky_relu()
  down1_pool <- down1 %>%
    layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2))
  # 64
  
  down2 <- down1_pool %>%
    layer_conv_2d(filters = 128, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation_leaky_relu()%>%
    layer_conv_2d(filters = 128, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation_leaky_relu()
  down2_pool <- down2 %>%
    layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2))
  # 32
  
  down3 <- down2_pool %>%
    layer_conv_2d(filters = 256, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation_leaky_relu()%>%
    layer_conv_2d(filters = 256, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation_leaky_relu()
  down3_pool <- down3 %>%
    layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2))
  # 16
  
  down4 <- down3_pool %>%
    layer_conv_2d(filters = 512, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation_leaky_relu()%>%
    layer_conv_2d(filters = 512, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation_leaky_relu()
  down4_pool <- down4 %>%
    layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2))
  # 8
  
  center <- down4_pool %>%
    layer_conv_2d(filters = 1024, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation_leaky_relu()%>%
    layer_conv_2d(filters = 1024, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation_leaky_relu()
  # center
  
  up4 <- center %>%
    layer_upsampling_2d(size = c(2, 2)) %>%
    {layer_concatenate(inputs = list(down4, .), axis = 3)} %>%
    layer_conv_2d(filters = 512, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation_leaky_relu()%>%
    layer_conv_2d(filters = 512, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation_leaky_relu()%>%
    layer_conv_2d(filters = 512, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation_leaky_relu()
  # 16
  
  up3 <- up4 %>%
    layer_upsampling_2d(size = c(2, 2)) %>%
    {layer_concatenate(inputs = list(down3, .), axis = 3)} %>%
    layer_conv_2d(filters = 256, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation_leaky_relu()%>%
    layer_conv_2d(filters = 256, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation_leaky_relu()%>%
    layer_conv_2d(filters = 256, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation_leaky_relu()
  # 32
  
  up2 <- up3 %>%
    layer_upsampling_2d(size = c(2, 2)) %>%
    {layer_concatenate(inputs = list(down2, .), axis = 3)} %>%
    layer_conv_2d(filters = 128, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation_leaky_relu()%>%
    layer_conv_2d(filters = 128, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation_leaky_relu()%>%
    layer_conv_2d(filters = 128, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation_leaky_relu()
  # 64
  
  up1 <- up2 %>%
    layer_upsampling_2d(size = c(2, 2)) %>%
    {layer_concatenate(inputs = list(down1, .), axis = 3)} %>%
    layer_conv_2d(filters = 64, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation_leaky_relu()%>%
    layer_conv_2d(filters = 64, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation_leaky_relu()%>%
    layer_conv_2d(filters = 64, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation_leaky_relu()
  # 128
  
  classify <- layer_conv_2d(up1,
                            filters = num_classes,
                            kernel_size = c(1, 1),
                            activation = "sigmoid")
  
  model <- keras_model(
    inputs = inputs,
    outputs = classify
  )
  return(model)
}

K <- backend()

epsilon = 1e-5
smooth = 1

#Binary Accuracy
acc_m <- custom_metric("acc_m", function(y_true, y_pred) {
  acc1 <- metric_binary_accuracy(y_true, y_pred)
  return(acc1)
})

#DICE Coefficient
dsc_m <- custom_metric("dsc_m", function(y_true, y_pred) {
  smooth = 1.
  y_true_f = k_flatten(y_true)
  y_pred_f = k_flatten(y_pred)
  intersection = k_sum(y_true_f * y_pred_f)
  score = (2. * intersection + smooth) / (k_sum(y_true_f) + k_sum(y_pred_f) + smooth)
  return(score)
})

#DICE Loss
dsc_l <- custom_metric("dsc_l", function(y_true, y_pred) {
  dsc_loss = 1 - dsc_m(y_true, y_pred)
  return(dsc_loss)
})

#Binary Cross Entropy Metric
bce_m <- custom_metric("bce_m", function(y_true, y_pred) {
  bce_metric <- metric_binary_crossentropy(y_true, y_pred)
  return(bce_metric)
})

#Binary Cross Entropy Loss
bce_l <- custom_metric("bce_l", function(y_true, y_pred) {
  bce_loss <- loss_binary_crossentropy(y_true, y_pred)
  return(bce_loss)
})

#BCE + DICE Loss
bce_dsc_l <- custom_metric("bce_dsc_l", function(y_true, y_pred) {
  bce_dsc_loss = bce_l(y_true, y_pred) + dsc_l(y_true, y_pred)
  return(bce_dsc_loss)
})

#Recall
recall_m <- custom_metric("recall_m", function(y_true, y_pred) {
  smooth=1
  y_pred_pos = k_clip(y_pred, 0, 1)
  y_pred_neg = 1 - y_pred_pos
  y_pos = k_clip(y_true, 0, 1)
  y_neg = 1 - y_pos
  tp = k_sum(y_pos * y_pred_pos)
  fp = k_sum(y_neg * y_pred_pos)
  fn = k_sum(y_pos * y_pred_neg) 
  prec = (tp + smooth)/(tp+fp+smooth)
  recall = (tp+smooth)/(tp+fn+smooth)
  return(recall)
})

#Precision
prec_m <- custom_metric("prec_m", function(y_true, y_pred) {
  smooth=1
  y_pred_pos = k_clip(y_pred, 0, 1)
  y_pred_neg = 1 - y_pred_pos
  y_pos = k_clip(y_true, 0, 1)
  y_neg = 1 - y_pos
  tp = k_sum(y_pos * y_pred_pos)
  fp = k_sum(y_neg * y_pred_pos)
  fn = k_sum(y_pos * y_pred_neg) 
  prec = (tp + smooth)/(tp+fp+smooth)
  recall = (tp+smooth)/(tp+fn+smooth)
  return(prec)
})

#Tversky Metric
tver <- custom_metric("tver", function(y_true, y_pred) {
  y_true_pos = k_flatten(y_true)
  y_pred_pos = k_flatten(y_pred)
  true_pos = k_sum(y_true_pos * y_pred_pos)
  false_neg = k_sum(y_true_pos * (1-y_pred_pos))
  false_pos = k_sum((1-y_true_pos)*y_pred_pos)
  alpha = 0.7
  tver = (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)
  return(tver)
})

#Tversky Loss
tver_l <- custom_metric("tver_l", function(y_true, y_pred) {
  tver_l =  1 - tver(y_true, y_pred)
  return(tver_l)
})

#Tversky Focal Loss
tver_lf <- custom_metric("tver_lf", function(y_true, y_pred) {
  pt_1 = tver(y_true, y_pred)
  gamma = 0.75
  return(k_pow((1-pt_1), gamma))
})

#Initiate model
model <- get_unet_128()

#Compile model
model %>% compile(
  optimizer = optimizer_adamax(), #Use ADAMAX Optimizer
  loss = dsc_l,
  metrics = list(tver_l, tver, dsc_m, bce_m, bce_dsc_l, acc_m, prec_m, recall_m)
)

#Reduce Learning Rate with Plateau callback
cb_plat <- callback_reduce_lr_on_plateau(monitor = "dsc_l",
                              factor = 0.1,
                              patience = 5,
                              verbose = 1,
                              min_delta = 1e-7,
                              mode = "min")

#Create checkpoint to write the generate weights to a folder
checkpoint_dir <- paste0("E:/topo_proj/scripts/checkpoints/10_14_2020", "/")
dir.create(checkpoint_dir, showWarnings = FALSE)
filepath <- file.path(checkpoint_dir, "weights.{epoch:02d}-{val_loss:.2f}.hdf5")

cb_check <- callback_model_checkpoint(
  filepath = filepath,
  save_weights_only = TRUE,
  verbose = 1,
  save_freq = "epoch",
)

#Create checkpoints to save metrics to a single CSV with update at end of each epoch
csv_nm <- paste0("E:/topo_proj/scripts/checkpoints/10_14_2020", ".csv")

cb_csv <- callback_csv_logger(csv_nm, separator=",")

#Fit model
model %>% fit(train_set,
	validation_data = test_set, 
	epochs = 100,
	callbacks = list(cb_check, cb_csv, cb_plat)
)

# Example of prediction on the validation data set

# Load the weights of a previous model
load_model_weights_hdf5(model, "E:/topo_proj/scripts/checkpoints/10_14_2020/weights.75-0.03.hdf5")

#Evaluate on different data sets
evaluate(model, test_set)
evaluate(model, val_set)
evaluate(model, oh_set)
evaluate(model, va_set)
evaluate(model, train_set)

val_re_df <- data.frame(set=character(), mod=character(), 
                        loss=numeric(), tver_l=numeric(), tver=numeric(), 
                    for (i in 1:length(mset)) {
  load_model_weights_hdf5(model, paste0(mpath, mset[i]))
  ky_r <- as.data.frame(evaluate(model, val_set))
  ky_r$set <- "KY"
  ky_r$mod <- mset[i]
  va_r <- as.data.frame(evaluate(model, va_set))
  va_r$set <- "VA"
  va_r$mod <- mset[i]
  oh_r <- as.data.frame(evaluate(model, oh_set))
  oh_r$set <- "OH"
  oh_r$mod <- mset[i]
  val_re_df <- rbind(val_re_df, ky_r, va_r, oh_r)
}
    dsc_m=numeric(), bce_dsc_l=numeric(), acc_m=numeric(),
                        prec_m=numeric(), recall_m=numeric())
mpath <- "E:/topo_proj/scripts/checkpoints/samp_size/"
mset <- list.files("E:/topo_proj/scripts/checkpoints/samp_size", pattern="\\.hdf5$")


val_re_df$size <- c(10, 10, 10, 10, 10, 15, 15, 15, 15, 15, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5)
write.csv(val_re_df, "E:/topo_proj/scripts/checkpoints/samp_size/sample_size_metrics.csv")



