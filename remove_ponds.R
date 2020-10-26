library(sf)
library(dplyr)

shp_lst <- list.files("D:/topo_check/va_topo_mines", pattern = ".shp$")

for (s in shp_lst) {
  input <- st_read(paste0("D:/topo_check/va_topo_mines", "/", s))
  output <- input %>% filter(FTR_TYPE != 'Settling Pond')
  output2 <- output %>% filter(FTR_TYPE != 'Tailings - Pond')
  st_write(output2, paste0("D:/topo_check/va_topo_mines2", "/", s))
}