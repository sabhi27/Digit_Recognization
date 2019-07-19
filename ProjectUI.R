  model_FileName <- "DigitRecognizer-Model.RData"

# Libraries
  library(shiny)
  library(EBImage)
  library(nnet)
  
  load(model_FileName)

  ui <- fluidPage(
    
    fileInput(inputId = "image",h3("Choose an Image"),multiple=TRUE,accept=c('image/png','image/jpeg')),
    textOutput("digit"),
    imageOutput("myImage")
  )
  server <- function(input, output) {
    
    output$digit <- renderText({
      inFile <- input$image
      if (is.null(inFile))
        return(NULL)
      
      old = inFile$datapath
      new = file.path(dirname(inFile$datapath),inFile$name)
      
      file.rename(from = old , to = new)
      inFile$datapath <- new
      
      Image <- readImage(inFile$datapath)
      nof=numberOfFrames(Image, type = c('total', 'render'))
      
      if(nof==1)
      {
        image=255*imageData(Image[1:28,1:28])
      }else 
        if(nof==3)
        {
          r=255*imageData(Image[1:28,1:28,1])
          g=255*imageData(Image[1:28,1:28,2])
          b=255*imageData(Image[1:28,1:28,3])
          
          image=0.21*r+0.72*g+0.07*b
          
          image=round(image)
        }
      image=t(image)
      
      image=as.vector(t(image))
      write.csv(t(as.matrix(image)),'threepx.csv',row.names = FALSE)
      test_FileName  <-'threepx.csv'
      new_Test_Dataset <- read.csv(test_FileName)    
      
      test_reduced <- new_Test_Dataset/255 
      
      test_reduced <- as.matrix(test_reduced) %*% prin_comp$rotation[,1:260]
      
      
      New_Predicted <- predict(model_final,test_reduced,type="class")
      paste("Predicted Value is",New_Predicted)
        })
  
    output$myImage <- renderImage({
      inFile <- input$image
      if (is.null(inFile))
        return(NULL)
    
      outfile <- tempfile(fileext='.jpg')
     
      old=inFile$datapat
      new = file.path(dirname(inFile$datapath),inFile$name)
      
      file.rename(from = old , to = new)
      inFile$datapath <- new
      
      list(src = inFile$datapath,
           contentType = 'image/jpeg',
           width =200,
           height=200,
           alt = "Predicted Image")
    }, deleteFile = TRUE)
  }
   shinyApp(ui = ui, server = server)
   
   
