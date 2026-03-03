
#' Group and Reorder Data by Clusters
#'
#' Rearranges the input count matrix and design matrix so that observations belonging
#' to the same cluster (e.g., patient) are contiguous. This is a required 
#' preprocessing step for block-wise variance estimation to ensure the C++ 
#' backend can iterate through clusters without jumping across memory.
#'
#' @param input_matrix A matrix-like object (e.g., sparse or dense count matrix) 
#' where columns represent individual cells/observations.
#' @param design_matrix A matrix of predictor variables where rows represent 
#' individual cells/observations.
#' @param clusters A vector of cluster or patient identifiers.
#'
#' @return A named list containing three elements:
#' \itemize{
#'   \item \code{input_matrix}: The reordered count matrix.
#'   \item \code{design_matrix}: The reordered design matrix.
#'   \item \code{clusters}: The reordered cluster vector, converted to numeric 
#'   indices based on the order of appearance.
#' }
#' 
#' @details 
#' The function converts the \code{clusters} vector into a factor based on its 
#' unique levels in order of appearance, then sorts all inputs based on these levels.
#' 
#' @export
#' @rawNamespace useDynLib(devil);
group_data <- function(input_matrix, design_matrix, clusters) {
  # 1. Validation: Ensure dimensions match
  if (ncol(input_matrix) != nrow(design_matrix)) {
    stop("Dimension mismatch: ncol(input_matrix) must equal nrow(design_matrix)")
  }
  if (length(clusters) != ncol(input_matrix)) {
    stop("Dimension mismatch: length(clusters) must match the number of cells")
  }
  
  # 2. Determine the sorting order
  # We convert to numeric based on unique levels to ensure the ordering is
  # stable and predictable for the block-wise C++ operations.
  clusters <- as.numeric(factor(clusters, levels = unique(clusters)))
  ord <- order(clusters)
  
  # 3. Rearrange all objects based on that order
  # input_matrix: Rearrange COLUMNS (cells)
  input_matrix_grouped <- input_matrix[, ord, drop = FALSE]
  
  # design_matrix: Rearrange ROWS (cells)
  design_matrix_grouped <- design_matrix[ord, , drop = FALSE]
  
  # clusters: Rearrange the vector itself
  clusters_grouped <- clusters[ord]
  
  # 4. Return as a structured list
  return(list(
    input_matrix = input_matrix_grouped,
    design_matrix = design_matrix_grouped,
    clusters = clusters_grouped
  ))
}