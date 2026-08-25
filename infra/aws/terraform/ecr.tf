resource "aws_ecr_repository" "controller" {
  name                 = "${var.name_prefix}/gpu-node-health-controller"
  image_tag_mutability = "IMMUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "controller" {
  repository = aws_ecr_repository.controller.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep the ten newest controller images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 10
      }
      action = {
        type = "expire"
      }
    }]
  })
}

resource "aws_ecr_repository" "training" {
  name                 = "${var.name_prefix}/distributed-training"
  image_tag_mutability = "IMMUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "training" {
  repository = aws_ecr_repository.training.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep the five newest CUDA training images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 5
      }
      action = {
        type = "expire"
      }
    }]
  })
}
