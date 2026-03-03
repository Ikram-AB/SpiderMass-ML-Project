library(shiny)
library(shinyjs)
library(reticulate)
library(DT)
library(plotly)
library(ggplot2)
library(jsonlite)

# ------------------------------------------------------------
# Python configuration
# ------------------------------------------------------------

# Use a specific Python executable 
use_python(Sys.which("python"), required = TRUE)

# Define project paths (more professional than hardcoding every file path)
project_root <- normalizePath("..")
src_dir <- file.path(project_root, "src")

# Load Python modules (updated file names)
source_python(file.path(src_dir, "spectrometry_utils.py"))
source_python(file.path(src_dir, "preprocessing.py"))
source_python(file.path(src_dir, "feature_engineering.py"))
source_python(file.path(src_dir, "model_training.py"))
source_python(file.path(src_dir, "realtime_inference_watcher.py"))

# Increase Shiny max upload size (in bytes)
options(shiny.maxRequestSize = 10000 * 1024^2)

# ------------------------------------------------------------
# User Interface
# ------------------------------------------------------------

ui <- fluidPage(
  useShinyjs(),
  tags$link(rel = "stylesheet", href = "https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css"),
  tags$link(rel = "stylesheet", href = "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css"),
  tags$link(rel = "stylesheet", href = "https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css"),

  tags$style(HTML("
    .navbar { background-color: #7C93C3 !important; }
    .navbar-nav { display: flex; flex-direction: row; }
    .nav-item { margin-right: 20px; }
    .nav-link { color: white !important; }
    .navbar-brand { color: white !important; margin-right: 20px; }
    .logo { height: 40px; margin-right: 10px; }
    .action-button-blue { background-color: #55679C; color: #ffffff; border-color: #55679C; }
    .action-button-blue:hover { background-color: #0056b3; color: #ffffff; border-color: #0056b3; }
    .icon-container { text-align: center; margin: 40px; }
    .icon { font-size: 50px; color: #007bff; margin: 10px; }
    .section { margin-bottom: 30px; }
    .section h3 { color: #7C93C3; }
  ")),

  tags$style(HTML("
    body {
      background-color: #FFFFFF;
      color: #55679C;
    }
    .sidebar-panel {
      background-color: #A0C4E2;
      color: #4F4F4F;
      border-right: 2px solid #4F4F4F;
      padding: 20px;
    }
    .main-panel {
      background-color:  #7C93C3;
      color: #7C93C3;
      padding: 20px;
    }
    hr {
      border: 2x solid #C0C0C0;
    }
  ")),

  navbarPage("Profiler",
    tabPanel("Home",
      fluidPage(
        titlePanel("Welcome to the Profiler Application"),
        div(class = "section",
          h3("Overview"),
          p("This application supports analysis and classification of mass spectrometry data. It can convert RAW files to mzML, load and preprocess data, train models, and run real-time classification.")
        ),
        div(class = "section",
          h3("Data Analysis"),
          p("In this tab you can:"),
          tags$ul(
            tags$li("Convert RAW files to mzML format."),
            tags$li("Load and preprocess data."),
            tags$li("Train and evaluate machine learning models.")
          )
        ),
        div(class = "section",
          h3("Real-Time Diagnosis"),
          p("In this tab you can:"),
          tags$ul(
            tags$li("Load trained models and the expected feature list."),
            tags$li("Monitor a directory and classify new files automatically.")
          )
        )
      )
    ),

    tabPanel("Data Analysis",
      fluidPage(
        sidebarLayout(
          sidebarPanel(
            h3("Conversion"),
            textInput("raw_files_path", "Enter Path for RAW Files Directory", value = ""),
            textInput("mzml_output_dir", "Enter Output Directory for mzML", value = ""),
            actionButton("convert_button", "Convert RAW to mzML", class = "action-button-blue"),
            textOutput("conversionResult"),
            verbatimTextOutput("debugConversion"),
            hr(),

            h3("Mean Spectra"),
            actionButton("plot_mean_spectra", "Plot Mean Spectra", class = "action-button-blue"),
            hr(),

            h3("Class Histogram"),
            actionButton("plot_class_histogram", "Plot Class Histogram", class = "action-button-blue"),
            hr(),

            h3("Data Loading"),
            uiOutput("paths_ui"),
            actionButton("add_path", "Add Path", class = "action-button-blue"),
            actionButton("load_data", "Load Data", class = "action-button-blue"),
            hr(),

            h3("Data Preprocessing"),
            actionButton("preprocess_data", "Preprocess Data", class = "action-button-blue"),
            hr(),

            h3("Train and Evaluate Models"),
            numericInput("n_components", "Number of SVD components", value = 50),
            selectInput("balance_method", "Balance Method", choices = c("None", "SMOTE", "ADASYN"), selected = "None"),
            actionButton("train_models", "Train and Evaluate Models", class = "action-button-blue"),
            hr(),

            h3("Save Model"),
            selectInput("model_name", "Select Model to Save", choices = NULL),
            textInput("model_save_folder", "Folder to Save Model", value = ""),
            actionButton("save_model", "Save Model", class = "action-button-blue")
          ),

          mainPanel(
            tabsetPanel(
              tabPanel("Conversion Result", textOutput("conversionResult")),
              tabPanel("Loaded Data", tableOutput("loaded_data")),
              tabPanel("Preprocessed Data", DTOutput("preprocessed_data_summary")),
              tabPanel("Resampled Data", DTOutput("resampled_data")),
              tabPanel("Model Scores", uiOutput("model_scores")),
              tabPanel("Confusion Matrices", uiOutput("confusion_matrices")),
              tabPanel("mzML Files", verbatimTextOutput("mzml_files")),
              tabPanel("Mean Spectra", plotOutput("mean_spectra_plot")),
              tabPanel("Class Histogram", plotOutput("class_histogram_plot"))
            )
          )
        )
      )
    ),

    tabPanel("Real-Time Diagnosis",
      sidebarLayout(
        sidebarPanel(
          textInput("watch_directory", "Watch Directory", value = ""),
          textInput("output_directory", "Output Directory", value = ""),
          actionButton("load_button", "Load Models and Features", class = "action-button-blue"),
          actionButton("start_button", "Start Watcher", class = "action-button-blue"),
          actionButton("stop_button", "Stop Watcher", class = "action-button-blue"),
          verbatimTextOutput("status")
        ),
        mainPanel(
          h3("Model Predictions"),
          tableOutput("prediction_table")
        )
      )
    )
  )
)

# ------------------------------------------------------------
# Server
# ------------------------------------------------------------

server <- function(input, output, session) {

  rv <- reactiveValues(data = NULL, preprocessed_data = NULL, resampled_data = NULL, pipelines = NULL, results = NULL)

  showLoadingModal <- function(message) {
    showModal(modalDialog(
      title = "Please wait",
      h5(paste0(message, "...")),
      footer = NULL,
      easyClose = FALSE
    ))
  }

  observeEvent(input$convert_button, {
    req(input$raw_files_path, input$mzml_output_dir)

    if (input$raw_files_path == "" || input$mzml_output_dir == "") {
      showModal(modalDialog(
        title = "Error",
        "Please specify both the RAW files directory and the mzML output directory.",
        easyClose = TRUE
      ))
      return(NULL)
    }

    showLoadingModal("Starting RAW to mzML conversion")

    tryCatch({
      # Function from spectrometry_utils.py
      convert_raw_to_mzml(input$raw_files_path, input$mzml_output_dir)
      removeModal()
      output$conversionResult <- renderText("Conversion completed successfully.")
      output$debugConversion <- renderPrint(cat("Conversion completed successfully.\n"))
    }, error = function(e) {
      removeModal()
      output$conversionResult <- renderText(paste("Conversion error:", e$message))
      output$debugConversion <- renderPrint(cat("Conversion error:", e$message, "\n"))
    })
  })

  observeEvent(input$add_path, {
    num_paths <- length(grep("^path", names(input)))
    insertUI(
      selector = "#paths_ui",
      ui = tagList(
        textInput(paste0("path", num_paths + 1), "Path to mzML Directory", placeholder = "Enter path here"),
        textInput(paste0("class", num_paths + 1), "Class Name for this Directory", placeholder = "Enter class name here")
      )
    )
  })

  observeEvent(input$load_data, {
    num_paths <- length(grep("^path", names(input)))
    paths <- sapply(1:num_paths, function(i) input[[paste0("path", i)]])
    classes <- sapply(1:num_paths, function(i) input[[paste0("class", i)]])

    paths <- paths[!is.null(paths) & paths != ""]
    classes <- classes[!is.null(classes) & classes != ""]

    if (length(paths) == 0 || length(classes) == 0 || length(paths) != length(classes)) {
      showModal(modalDialog(
        title = "Error",
        "Please fill in all path and class fields.",
        easyClose = TRUE
      ))
      return(NULL)
    }

    showLoadingModal("Loading data")

    tryCatch({
      # Function from spectrometry_utils.py
      df <- load_data(paths, classes)
      rv$data <- df
      removeModal()
      output$loaded_data <- renderTable(head(df))
    }, error = function(e) {
      removeModal()
      showModal(modalDialog(
        title = "Error",
        paste("Loading error:", e$message),
        easyClose = TRUE
      ))
    })
  })

  observeEvent(input$preprocess_data, {
    req(rv$data)

    showLoadingModal("Preprocessing data")

    tryCatch({
      # Function from preprocessing.py (or spectrometry_utils.py if you kept it there)
      df <- preprocess_data(rv$data)
      rv$preprocessed_data <- df

      output$preprocessed_data_summary <- renderDT(datatable(df))
      removeModal()

      showModal(modalDialog(
        title = "Success",
        "Data preprocessing completed successfully.",
        easyClose = TRUE
      ))
    }, error = function(e) {
      removeModal()
      showModal(modalDialog(
        title = "Error",
        paste("Preprocessing error:", e$message),
        easyClose = TRUE
      ))
    })
  })

  observeEvent(input$plot_class_histogram, {
    req(rv$data)

    output$class_histogram_plot <- renderPlot({
      ggplot(rv$data, aes(x = factor(Class))) +
        geom_bar(fill = "steelblue") +
        labs(title = "Class Distribution", x = "Class", y = "Number of samples") +
        theme_minimal()
    })
  })

  observeEvent(input$train_models, {
    req(rv$preprocessed_data)

    showLoadingModal("Training and evaluating models")

    tryCatch({
      # Apply optional resampling (feature_engineering.py)
      if (input$balance_method == "SMOTE") {
        df_resampled <- apply_smote(rv$preprocessed_data)
      } else if (input$balance_method == "ADASYN") {
        df_resampled <- apply_adasyn(rv$preprocessed_data)
      } else {
        df_resampled <- rv$preprocessed_data
      }

      # Output folder for confusion matrices
      output_dir <- "C:/Users/Admin/Documents/Ikram_Rshiny/R shiny/matrice_confusion_output"
      dir.create(output_dir, showWarnings = FALSE)

      # Train and evaluate (model_training.py)
      results <- train_and_evaluate_pipeline(df_resampled, input$n_components, output_dir)

      rv$pipelines <- results[[1]]
      rv$results <- results[[3]]

      removeModal()
      showModal(modalDialog(
        title = "Success",
        "Models trained and evaluated successfully.",
        easyClose = TRUE
      ))
    }, error = function(e) {
      removeModal()
      showModal(modalDialog(
        title = "Error",
        paste("Training error:", e$message),
        easyClose = TRUE
      ))
    })
  })

  # Real-time: load models + features
  observeEvent(input$load_button, {
    model_paths <- c(
      "C:/Users/Admin/Documents/Ikram_Rshiny/R shiny/HistGradientBoostingClassifier_last.pkl",
      "C:/Users/Admin/Documents/Ikram_Rshiny/R shiny/RandomForestClassifier_last.pkl"
    )

    nf_path <- "C:/Users/Admin/Documents/Ikram_Rshiny/R shiny/numeric_features.pkl"

    # Functions from realtime_inference_watcher.py
    py$load_models_and_features(model_paths, nf_path)
    output$status <- renderText("Models and features loaded.")
  })

  # Start watcher
  observeEvent(input$start_button, {
    req(input$watch_directory, input$output_directory)
    py$start_watcher(input$watch_directory, input$output_directory)
    output$status <- renderText("Watcher started.")
  })

  # Stop watcher
  observeEvent(input$stop_button, {
    py$stop_watcher()
    output$status <- renderText("Watcher stopped.")
  })
}

shinyApp(ui, server)