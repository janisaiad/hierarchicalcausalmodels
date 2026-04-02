#
# This is the server logic of a Shiny web application. You can run the 
# application by clicking 'Run App' above.
#
# Find out more about building applications with Shiny here:
# 
#    http://shiny.rstudio.com/
#

library(shiny)
library(ggplot2)
library(grid)
library(gridExtra)

# Define server logic required to draw a histogram
shinyServer(function(input, output) {
  
  output$distPlot <- renderPlot({
    
    # generate bins based on input$bins from ui.R
    star <- read.table("STAR_Students.tab",sep="\t", header=TRUE)
    
    # Keep the columns only relevant to the first grade students
    columns <- c("gender","g1classtype","g1schid","g1surban","g1tchid","g1tgen","g1trace","g1thighdegree","g1tcareer","g1tyears","g1tmathss")
    data <- star[,columns]
    colnames(data)[11] <- "MathScore"
    #x    <- data$g1tmathss[!is.na(data$g1tmathss)]
    #bins <- seq(min(x), max(x), length.out = input$bins + 1)
    
    # draw the histogram with the specified number of bins
    #hist(x, breaks = bins, col = 'darkgray', border = 'white', xlab= "Math Scores", main ="1st grade student math scores")
    plot1 <- ggplot(data, aes(x = MathScore)) + 
      geom_histogram(bins = input$bins , fill = "black", col = "grey") + ggtitle("1st Grade Student Math Scores")+
      scale_x_continuous(breaks = seq(0, 700, by = 25))

    grid.arrange(plot1)
    
  })
  
})
