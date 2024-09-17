library(raster)
library(sf)
library(lattice)
library(terra)
library(rasterVis)
library(RColorBrewer)
library(maps)
library(ggplot2)

#### Set the working directory
setwd("C:/Users/gstel/OneDrive/Desktop/IITA/Soybean Rust/Project/Soybean Rust severity")

###Load the raster stack
ras<- rast("Raster_stack_AllSBR.tif")
names(ras)

### Load the national boundaries
aoi <- st_read("AOI_BoundariesPRJ.shp")
aoi <- st_transform(aoi, crs(ras))
aoi_sp <- as(aoi, "Spatial")
# Function to convert SpatialPolygons to coordinates
convert_to_coords <- function(sp_poly) {
  polys <- slot(sp_poly, "polygons")
  coords <- matrix(nrow = 0, ncol = 2)
  for (i in seq_along(polys)) {
    for (j in seq_along(slot(polys[[i]], "Polygons"))) {
      crds <- rbind(slot(slot(polys[[i]], "Polygons")[[j]], "coords"), c(NA, NA))
      coords <- rbind(coords, crds)
    }
  }
  coords
}

# Convert AOI to coordinate matrix
coords <- convert_to_coords(aoi_sp)
coords[, 1] <- coords[, 1] + 180 # Adjust coordinates if needed
coords[, 2] <- coords[, 2] + 90  # Adjust coordinates if needed

### Extract precipitation only
prec.names <- c("prec2012JunJul","prec2012NovDec","prec2012DecJan","prec2013JunJul","prec2013NovDec","prec2013DecJan",
                "prec2014JunJul","prec2014NovDec","prec2014DecJan","prec2015JunJul","prec2015NovDec","prec2015DecJan",
                "prec2016JunJul","prec2016NovDec","prec2016DecJan","prec2017JunJul","prec2017NovDec","prec2017DecJan",
                "prec2018JunJul","prec2018NovDec","prec2018DecJan","prec2019JunJul","prec2019NovDec","prec2019DecJan",
                "prec2020JunJul","prec2020NovDec","prec2020DecJan","prec2022JunJul","prec2022NovDec","prec2022DecJan",
                "prec2023JunJul","prec2023NovDec","prec2023DecJan")
prec <- ras[[prec.names]]

## Calculate the average conditions
avgPrec <- mean(prec)
plot(avgPrec)

## Calculate how much each year deviates from the average conditions
year2012JunJul <- prec[[1]] - avgPrec
year2012NovDec <- prec[[2]] - avgPrec
year2012DecJan <- prec[[3]] - avgPrec
year2013JunJul <- prec[[4]] - avgPrec
year2013NovDec <- prec[[5]] - avgPrec
year2013DecJan <- prec[[6]] - avgPrec
year2014JunJul <- prec[[7]] - avgPrec
year2014NovDec <- prec[[8]] - avgPrec
year2014DecJan <- prec[[9]] - avgPrec
year2015JunJul <- prec[[10]] - avgPrec
year2015NovDec <- prec[[11]] - avgPrec
year2015DecJan <- prec[[12]] - avgPrec
year2016JunJul <- prec[[13]] - avgPrec
year2016NovDec <- prec[[14]] - avgPrec
year2016DecJan <- prec[[15]] - avgPrec
year2017JunJul <- prec[[16]] - avgPrec
year2017NovDec <- prec[[17]] - avgPrec
year2017DecJan <- prec[[18]] - avgPrec
year2018JunJul <- prec[[19]] - avgPrec
year2018NovDec <- prec[[20]] - avgPrec
year2018DecJan <- prec[[21]] - avgPrec
year2019JunJul <- prec[[22]] - avgPrec
year2019NovDec <- prec[[23]] - avgPrec
year2019DecJan <- prec[[24]] - avgPrec
year2020JunJul <- prec[[25]] - avgPrec
year2020NovDec <- prec[[26]] - avgPrec
year2020DecJan <- prec[[27]] - avgPrec
year2022JunJul <- prec[[28]] - avgPrec
year2022NovDec <- prec[[29]] - avgPrec
year2022DecJan <- prec[[30]] - avgPrec
year2023JunJul <- prec[[31]] - avgPrec
year2023NovDec <- prec[[32]] - avgPrec
year2023DecJan <- prec[[33]] - avgPrec

deviation_average <- c(year2012JunJul, year2012NovDec, year2012DecJan, year2013JunJul, year2013NovDec, 
  year2013DecJan, year2014JunJul, year2014NovDec, year2014DecJan, year2015JunJul, 
  year2015NovDec, year2015DecJan, year2016JunJul, year2016NovDec, year2016DecJan, 
  year2017JunJul, year2017NovDec, year2017DecJan, year2018JunJul, year2018NovDec, 
  year2018DecJan, year2019JunJul, year2019NovDec, year2019DecJan, year2020JunJul, 
  year2020NovDec, year2020DecJan, year2022JunJul, year2022NovDec, year2022DecJan, 
  year2023JunJul, year2023NovDec, year2023DecJan)


names(deviation_average)

clorpalet <- colorRampPalette(brewer.pal(10,"RdYlBu"))
breaks <- c(-Inf,-400,-300,-200,-100, 0, 100, 200, 300, 400,Inf)

# Custom panel function to add AOI outline
add_AOI <- function(...) {
  panel.levelplot(...)
  sp.polygons(aoi_sp, fill = NA, col = "black", lwd = 2)  # Add AOI outline
}

png(file = "LongtermRainfallDeviations_SEA_SBr_with_AOI.png", width = 11000, height = 9000, units = "px", res = 650, type = "cairo")
levelplot(deviation_average, at = breaks, col.regions = clorpalet(10),
          panel = function(...) {
            panel.levelplot(...)
            panel.polygon(coords, border = "black", lwd = 1.5)  # Add AOI outline
          },
          margin = FALSE)
dev.off()

##### %%%%%%%%%%%%%%%%%%%%%%%%%% THE END %%%%%%%%%%%%%%%%%%%%%%%%% #####




