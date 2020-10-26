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
va2 <- va_2


train_2x <- rbind(train, train_b2)
set.seed(42)
rows <- sample(nrow(train_2x))
train2x <- train_2x[rows,]
quad2 <- as.data.frame(str_split(train2x$quad, "/", simplify=TRUE))
train2x$quad2 <- quad2$V6

chp_cnt <- train2x %>% group_by(quad2) %>% count()
chp_cnt2 <- as.data.frame(chp_cnt)
chp_cnt2$n2 <- chp_cnt2$n - 150
chp_cnt2$wght <- chp_cnt2$n2/max(chp_cnt2$n2)

train2xx <- train2x

set.seed(42)
samp2 <- chp_cnt2 %>% sample_n(2, weight = wght)
train2af <- train2xx %>% filter(quad2 %in% samp2$quad2)
train2a <- train2af[,1:2]

set.seed(47)
samp2 <- chp_cnt2 %>% sample_n(2, weight = wght)
train2bf <- train2xx %>% filter(quad2 %in% samp2$quad2)
train2b <- train2bf[,1:2]

set.seed(52)
samp2 <- chp_cnt2 %>% sample_n(2, weight = wght)
train2cf <- train2xx %>% filter(quad2 %in% samp2$quad2)
train2c <- train2cf[,1:2]

set.seed(57)
samp2 <- chp_cnt2 %>% sample_n(2, weight = wght)
train2df <- train2xx %>% filter(quad2 %in% samp2$quad2)
train2d <- train2df[,1:2]

set.seed(62)
samp2 <- chp_cnt2 %>% sample_n(2, weight = wght)
train2ef <- train2xx %>% filter(quad2 %in% samp2$quad2)
train2e <- train2ef[,1:2]

set.seed(42)
samp5 <- chp_cnt2 %>% sample_n(5, weight = wght)
train5af <- train2xx %>% filter(quad2 %in% samp5$quad2)
train5a <- train5af[,1:2]

set.seed(47)
samp5 <- chp_cnt2 %>% sample_n(5, weight = wght)
train5bf <- train2xx %>% filter(quad2 %in% samp5$quad2)
train5b <- train5bf[,1:2]

set.seed(52)
samp5 <- chp_cnt2 %>% sample_n(5, weight = wght)
train5cf <- train2xx %>% filter(quad2 %in% samp5$quad2)
train5c <- train5cf[,1:2]

set.seed(57)
samp5 <- chp_cnt2 %>% sample_n(5, weight = wght)
train5df <- train2xx %>% filter(quad2 %in% samp5$quad2)
train5d <- train5df[,1:2]

set.seed(62)
samp5 <- chp_cnt2 %>% sample_n(5, weight = wght)
train5ef <- train2xx %>% filter(quad2 %in% samp5$quad2)
train5e <- train5ef[,1:2]


set.seed(42)
samp10 <- chp_cnt2 %>% sample_n(10, weight = wght)
train10af <- train2xx %>% filter(quad2 %in% samp10$quad2)
train10a <- train10af[,1:2]

set.seed(47)
samp10 <- chp_cnt2 %>% sample_n(10, weight = wght)
train10bf <- train2xx %>% filter(quad2 %in% samp10$quad2)
train10b <- train10bf[,1:2]

set.seed(52)
samp10 <- chp_cnt2 %>% sample_n(10, weight = wght)
train10cf <- train2xx %>% filter(quad2 %in% samp10$quad2)
train10c <- train10cf[,1:2]

set.seed(57)
samp10 <- chp_cnt2 %>% sample_n(10, weight = wght)
train10df <- train2xx %>% filter(quad2 %in% samp10$quad2)
train10d <- train10df[,1:2]

set.seed(62)
samp10 <- chp_cnt2 %>% sample_n(10, weight = wght)
train10ef <- train2xx %>% filter(quad2 %in% samp10$quad2)
train10e <- train10ef[,1:2]


set.seed(42)
samp15 <- chp_cnt2 %>% sample_n(15, weight = wght)
train15af <- train2xx %>% filter(quad2 %in% samp15$quad2)
train15a <- train15af[,1:2]

set.seed(47)
samp15 <- chp_cnt2 %>% sample_n(15, weight = wght)
train15bf <- train2xx %>% filter(quad2 %in% samp15$quad2)
train15b <- train15bf[,1:2]

set.seed(52)
samp15 <- chp_cnt2 %>% sample_n(15, weight = wght)
train15cf <- train2xx %>% filter(quad2 %in% samp15$quad2)
train15c <- train15cf[,1:2]

set.seed(57)
samp15 <- chp_cnt2 %>% sample_n(15, weight = wght)
train15df <- train2xx %>% filter(quad2 %in% samp15$quad2)
train15d <- train15df[,1:2]

set.seed(62)
samp15 <- chp_cnt2 %>% sample_n(15, weight = wght)
train15ef <- train2xx %>% filter(quad2 %in% samp15$quad2)
train15e <- train15ef[,1:2]




set.seed(42)
samp20 <- chp_cnt2 %>% sample_n(20, weight = wght)
train20af <- train2xx %>% filter(quad2 %in% samp20$quad2)
train20a <- train20af[,1:2]

set.seed(47)
samp20 <- chp_cnt2 %>% sample_n(20, weight = wght)
train20bf <- train2xx %>% filter(quad2 %in% samp20$quad2)
train20b <- train20bf[,1:2]

set.seed(52)
samp20 <- chp_cnt2 %>% sample_n(20, weight = wght)
train20cf <- train2xx %>% filter(quad2 %in% samp20$quad2)
train20c <- train20cf[,1:2]

set.seed(57)
samp20 <- chp_cnt2 %>% sample_n(20, weight = wght)
train20df <- train2xx %>% filter(quad2 %in% samp20$quad2)
train20d <- train20df[,1:2]

set.seed(62)
samp20 <- chp_cnt2 %>% sample_n(20, weight = wght)
train20ef <- train2xx %>% filter(quad2 %in% samp20$quad2)
train20e <- train20ef[,1:2]




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
create_dataset <- function(data, train, batch_size =32L, buff_size=1000) {

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
    dataset_shuffle(buffer_size = buff_size)
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
                              mode = "max")

#Create checkpoint to write the generate weights to a folder
checkpoint_dir <- paste0("E:/topo_proj/scripts/checkpoints/10_3_2020c", "/")
dir.create(checkpoint_dir, showWarnings = FALSE)
filepath <- file.path(checkpoint_dir, "weights.{epoch:02d}-{val_loss:.2f}.hdf5")

cb_check <- callback_model_checkpoint(
  filepath = filepath,
  save_weights_only = TRUE,
  verbose = 1,
  save_freq = "epoch",
)

#Create checkpoints to save metrics to a single CSV with update at end of each epoch
csv_nm <- paste0("E:/topo_proj/scripts/checkpoints/10_3_2020c", ".csv")

cb_csv <- callback_csv_logger(csv_nm, separator=",")

#Fit model
model %>% fit(train_set,
	validation_data = test_set, 
	epochs = 100,
	callbacks = list(cb_check, cb_csv, cb_plat)
)

# Example of prediction on the validation data set

# Load the weights of a previous model
load_model_weights_hdf5(model, "E:/topo_proj/scripts/checkpoints/10_3_2020c/weights.95-0.03.hdf5")

#Evaluate on different data sets
evaluate(model, test_set)
evaluate(model, val_set)
evaluate(model, oh_set)
evaluate(model, va_set)
evaluate(model, train_set)

set.seed(42)
rows <- sample(nrow(val))
valx <- val[rows,]

val_setx <- create_dataset(valx, train=FALSE)

#Create batch from validation set
batch <- val_setx %>% as_iterator() %>% iter_next()
predictions <- predict(model, batch)

#Display image, mask, and prediction
images <- tibble(
  image = batch[[1]] %>% array_branch(1),
  predicted_mask = predictions[,,,1] %>% array_branch(1),
  mask = batch[[2]][,,,1]  %>% array_branch(1)
) %>% 
  sample_n(32) %>% 
  map_depth(2, function(x) {
    as.raster(x) %>% magick::image_read() %>% magick::image_normalize()
  }) %>% 
  map(~do.call(c, .x))
out <- magick::image_append(c(
  magick::image_append(images$mask, stack = TRUE),
  magick::image_append(images$image, stack = TRUE), 
  magick::image_append(images$predicted_mask, stack = TRUE)
  )
)

#Visualize and save 32 predictions
print(out)
image_write(out, "E:/example_results.png")

#Function to add rowsand columns to data to generate image chips of fixed size
modify_raster_margins <- function(x,extent_delta=c(0,0,0,0),value=NA)
{
  x_extents <- extent(x)
  res_x <- res(x)
  
  x_modified <- x
  
  if(any(extent_delta < 0))
  {
    # Need to crop
    # ul:
    ul_mod <- extent_delta[c(1,3)] * res_x
    ul_mod[ul_mod > 0] <- 0
    lr_mod <- extent_delta[c(2,4)] * res_x
    lr_mod[lr_mod > 0] <- 0
    # This works fine, but for some reason CRAN doesn't like it:	
    #	crop_extent <- as.vector(x_extents)
    crop_extent <- c(x_extents@xmin,x_extents@xmax,x_extents@ymin,x_extents@ymax)
    crop_extent[c(1,3)] <- crop_extent[c(1,3)] - ul_mod
    crop_extent[c(2,4)] <- crop_extent[c(2,4)] + lr_mod
    
    x_modified <- crop(x_modified,crop_extent)
  }
  
  if(any(extent_delta > 0))
  {
    # Need to crop
    # ul:
    ul_mod <- extent_delta[c(1,3)] * res_x
    ul_mod[ul_mod < 0] <- 0
    lr_mod <- extent_delta[c(2,4)] * res_x
    lr_mod[lr_mod < 0] <- 0
    #		Again, a hack for CRAN?		
    #		extend_extent <- as.vector(x_extents)
    extend_extent <- c(x_extents@xmin,x_extents@xmax,x_extents@ymin,x_extents@ymax)
    extend_extent[c(1,3)] <- extend_extent[c(1,3)] - ul_mod
    extend_extent[c(2,4)] <- extend_extent[c(2,4)] + lr_mod
    
    x_modified <- extend(x_modified,extend_extent,value=value)
  }
  
  return(x_modified)
}

#List of topos to process
new_topos_test <- as.data.frame(topo_folders_test$path)
names(new_topos_test) <- c("path")
new_topos_test$set <- "test"
new_topos_test$base <- basename(new_topos_test$path)
new_topos_test$topo_folder <- paste0("E:/topo_proj/topo_check/processing/ky_topo8bit/", new_topos_test$base, ".png")
new_topos_test$mine_folder <- paste0("E:/topo_proj/topo_check/ky_topos_mines_checked2/", new_topos_test$base, ".shp")
new_topos_test$mask_folder <- paste0("E:/topo_proj/topo_check/ky_topo_quads/", new_topos_test$base, ".shp")

new_topos_val <- as.data.frame(topo_folders_val$path)
names(new_topos_val) <- c("path")
new_topos_val$set <- "val"
new_topos_val$base <- basename(new_topos_val$path)
new_topos_val$topo_folder <- paste0("E:/topo_proj/topo_check/processing/ky_topo8bit/", new_topos_val$base, ".png")
new_topos_val$mine_folder <- paste0("E:/topo_proj/topo_check/ky_topos_mines_checked2/", new_topos_val$base, ".shp")
new_topos_val$mask_folder <- paste0("E:/topo_proj/topo_check/ky_topo_quads/", new_topos_val$base, ".shp")

new_topos_oh <- as.data.frame(oh_folders)
names(new_topos_oh) <- c("path")
new_topos_oh$set <- "oh"
new_topos_oh$base <- basename(new_topos_oh$path)
new_topos_oh$topo_folder <- paste0("E:/topo_proj/topo_check/processing/oh_topo8bit/", new_topos_oh$base, ".png")
new_topos_oh$mine_folder <- paste0("E:/topo_proj/topo_check/oh_topo_mines2/", new_topos_oh$base, ".shp")
new_topos_oh$mask_folder <- paste0("E:/topo_proj/topo_check/oh_topo_quads/", new_topos_oh$base, ".shp")

new_topos_va <- as.data.frame(va_folders)
names(new_topos_va) <- c("path")
new_topos_va$set <- "va"
new_topos_va$base <- basename(new_topos_va$path)
new_topos_va$topo_folder <- paste0("E:/topo_proj/topo_check/processing/va_topo8bit/", new_topos_va$base, ".png")
new_topos_va$mine_folder <- paste0("E:/topo_proj/topo_check/va_topo_mines2/", new_topos_va$base, ".shp")
new_topos_va$mask_folder <- paste0("E:/topo_proj/topo_check/va_topo_quads/", new_topos_va$base, ".shp")

new_topos <- rbind(new_topos_test, new_topos_val, new_topos_oh, new_topos_va)

output_directory <- "E:/topo_proj/topo_check/predictions/"

#Generate blank data frame to store assessment metrics
metrics_df <- data.frame(set=character(), quad=character(), acc=numeric(), recall=numeric(), precision=numeric(), f1=numeric(), specificity=numeric())

library(sf)
library(raster)
library(fasterize)

#Loop through  the topo maps 
for (t in 1:nrow(new_topos)) {
  
  #Input topo, mines, and extent (Match all projections to the topo map)
  
  topo <- brick(new_topos$topo_folder[t])
  mask <- st_read(new_topos$mask_folder[t])
  mask_cr <- st_transform(mask, crs=crs(topo))
  mine <- st_read(new_topos$mine_folder[t])
  mine_cr <- st_transform(mine, crs=crs(topo))
  
  #Crop topo to quad extent
  topo_mask <- crop(topo, mask_cr)
  
  #Gather info to generate predictions using subset arrays of defined size and overlap
  across_cnt = ncol(topo_mask)
  down_cnt = nrow(topo_mask)
  tile_size_across = 128
  tile_size_down = 128
  overlap_across = 64
  overlap_down = 64
  across <- ceiling(across_cnt/overlap_across)
  down <- ceiling(down_cnt/overlap_down)
  across_add <- (across*overlap_across)-across_cnt 
  down_add <- (down*overlap_down)-down_cnt 
  topo_e <- modify_raster_margins(topo_mask,extent_delta=c(0,across_add,0,down_add),value=0)
  across_seq <- seq(0, across-2, by=1)
  down_seq <- seq(0, down-2, by=1)
  across_seq2 <- (across_seq*overlap_across)+1
  down_seq2 <- (down_seq*overlap_down)+1
  
  #Generate empty array to save predictions into
  topo_na <- raster(topo_e[[1]])
  topo_na[] <- NA
  full_array <- as.array(topo_na)
  
  #Copy topo to array
  topo_ae <- as.array(topo_e)
  
  #Loop through row/column combinations to make predictions for entire image 
  for (c in across_seq2){
    for (r in down_seq2){
      t1 <- topo_ae
      c1 <- c
      r1 <- r
      c2 <- c + 127
      r2 <- r + 127
      t2 <- as.array(t1[r1:r2, c1:c2, 1:3])
      t3 <- as.array(t2)/255
      t4 <- array(NA, dim=c(1, 128, 128, 3))
      t4[1,,,] <- t3
      t5 <- predict(model, t4)
      t6 <- as.array(t5[1,22:104,22:104,1])
      start_c <- c1+21
      stop_c <- c1+103
      start_r <- r1+21
      stop_r <- r1+103
      full_array[start_r:stop_r, start_c:stop_c, 1] <- t6
    }
  }
  
  #Convert array to raster and define projection info
  topo_pred <- brick(full_array)
  topo_pred@extent <- topo_e@extent
  topo_pred@crs <- topo_e@crs
  
  #Write result to file
  writeRaster(topo_pred, paste0(output_directory, new_topos$base[t], ".tif"))
  
  #Create assessment metrics and write to table
  predG <- topo_pred > .5
  blankG <- topo_pred
  blankG[] <- NA
  trueG <- fasterize(mine_cr, predG, background=0)
  trueGb <- (trueG+1)*10
  comp1 <- predG+trueGb
  comp2 <- crop(comp1, extent(comp1, 128,nrow(comp1)-128, 128, ncol(comp1)-128))
  table1 <- freq(comp2)
  recall_calc <- table1[4,2]/(table1[4,2]+table1[3,2])
  precision_calc <- table1[4,2]/(table1[4,2]+table1[2,2])
  spec_calc <- table1[1,2]/(table1[1,2]+table1[2,2])
  acc_calc <- (table1[4,2]+table1[1,2])/(table1[1,2]+table1[2,2]+table1[3,2]+table1[4,2])
  f1_calc <- (2*precision_calc*recall_calc)/(precision_calc+recall_calc)
  case_metrics <- data.frame(set=new_topos$set[t], quad=new_topos$base[t], acc=acc_calc, recall=recall_calc, precision=precision_calc, f1=f1_calc, specificity=spec_calc)
  metrics_df <- rbind(metrics_df, case_metrics)
  print(paste0("Completed for ", t))
}

write.csv(metrics_df, paste0(output_directory, "metrics2.csv"))










#Combine all tibbles into a list
train_lst <- list(train2a, train2b, train2c, train2d, train2e,
                  train5a, train5b, train5c, train5d, train5e, 
                  train10a, train10b, train10c, train10d, train10e,
                  train15a, train15b, train15c, train15d, train15e,
                  train20a, train20b, train20c, train20d, train20e)
#Generate names for tibles in list
train_nms <- c("train2a", "train2b", "train2c", "train2d", "train2e", 
               "train5a", "train5b", "train5c", "train5d", "train5e", 
               "train10a", "train10b", "train10c", "train10d", "train10e",
               "train15a", "train15b", "train15c", "train15d", "train15e",
               "train20a", "train20b", "train20c", "train20d", "train20e") 
               
names(train_lst) <- train_nms

checkpoint_dir <- paste0("E:/topo_proj/scripts/checkpoints/samp_size", "/")
dir.create(checkpoint_dir, showWarnings = FALSE)

for (m in 1:length(train_lst)) {
  train_in <- as_tibble(as.data.frame(train_lst[m]))
  mname <- names(train_lst[m])
  names(train_in) <- c("img", "mask")

  training_dataset <- create_dataset(train_in, train = TRUE, buff_size=nrow(train_in))
  
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
  
  filepath <- paste0(checkpoint_dir, mname, ".hdf5")

  cb_check <- callback_model_checkpoint(
    filepath = filepath,
    save_best_only=TRUE,
    mode="min",
    save_weights_only = TRUE,
    verbose = 1,
    save_freq = "epoch",
  )

  #Create checkpoints to save metrics to a single CSV with update at end of each epoch
  csv_nm <- paste0("E:/topo_proj/scripts/checkpoints/samp_size/", mname, ".csv")

  cb_csv <- callback_csv_logger(csv_nm, separator=",")

  #Fit model
  model %>% fit(training_dataset,
                validation_data = test_set, 
                epochs = 100,
                callbacks = list(cb_check, cb_csv, cb_plat)
  )
}



