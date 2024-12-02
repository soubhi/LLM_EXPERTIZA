# LLM Codebase (Expertiza) Vectorizer

## Overview
This project is designed to read, embed, and analyze code files using CodeBERT and other machine learning tools. The embeddings and metadata are stored in MongoDB, and FAISS is used for similarity search. The project also leverages OpenAI's API for analyzing code relationships and structure.

## Features
- **Code Embedding with CodeBERT**: Tokenizes and embeds code using `microsoft/graphcodebert-base`.
- **MongoDB Integration**: Stores code metadata and embeddings for efficient querying.
- **FAISS Integration**: Uses FAISS to search for similar code embeddings.
- **OpenAI API Integration**: Generates insights about code relationships and design using GPT-4.

## Requirements
The following Python libraries are required to run the project:
- `pymongo==4.5.0`
- `numpy==1.24.3`
- `torch==2.0.1`
- `faiss-cpu==1.7.3`  *(Use `faiss-gpu` if GPU support is needed)*
- `openai==0.11.0`
- `transformers==4.33.0`
- `python-dotenv`

## Installation
1. **Clone the repository locally to get root directory for input(Expertiza)**:
   ```bash
   git clone https://github.com/JackLiu28/LLM_EXPERTIZA.git
   cd LLM_EXPERTIZA
   ```
   *Note that any repository Ruby Codebase can be analyzed with this tool
   * This would be the input that is vectorized into the LLM for processing
3. **Requirement.txt install**:
    ```bash 
    pip install -r requirements.txt          
4. **setup environment file**:
    ```
    OMP_NUM_THREADS=1
    OPENAI_API_KEY={API Key}
    MONGODB_URL={MongoDB connection url}
    CODEBASE_PATH={path to codebase repository}
5. **run the script**:
     ```bash
     C:.../anaconda3/envs/pyml/python.exe .../LLM_EXPERTIZA/LLM_VECTOR.py
     
6. **Output**
   LLM_output.txt file EX: 
```bash
The code provided shows ActiveRecord models in a Ruby on Rails application. In general, the code is well-structured and adheres to Rails conventions. However, there are a few design and structural improvements that can be made to enhance the code maintainability, reduce coupling, and improve readability.

1. Use of `dependent` in associations: In the `CalculatedPenalty` model, the `:participant` association has a `dependent: :destroy` option. This means that if a `CalculatedPenalty` object is destroyed, its associated `Participant` object would be destroyed too. It's essential to ensure this is the desired behavior, as it could lead to data loss. It seems more reasonable that the `Participant` record should be left untouched when a `CalculatedPenalty` is removed.

    ```ruby
    class CalculatedPenalty < ApplicationRecord
      belongs_to :participant, class_name: 'Participant', foreign_key: 'participant_id'
      belongs_to :deadline_type, class_name: 'DeadlineType', foreign_key: 'deadline_type_id'
    end
    ```

2. Possible Polymorphic Association: If you find that `AssignmentDueDate`, and `TopicDueDate` shares much common behavior, consider refactoring the `parent_id` to a polymorphic relationship. This would involve adding a `parent_type` field and standardizing the `parent_id` field across models that behave similarly.

    ```ruby
    class DueDate < ApplicationRecord
      belongs_to :parent, polymorphic: true
      belongs_to :deadline_type, class_name: 'DeadlineType', foreign_key: 'deadline_type_id'
    end
    ```
    ```ruby
    class Assignment < ApplicationRecord
      has_many :due_dates, as: :parent
    end
    ```

3. Interface Segregation Principle: `DeadlineType` class is managing relationships for both `assignment_due_dates` and `topic_due_dates`. It might violate Interface segregation principle as `DeadlineType` has to depend on both `AssignmentDueDate` and `TopicDueDate`. To solve it, it would be better if each class manages its own dependencies. We are also creating a scope here named `assignment_due_dates` which brings all the records.

    ```ruby
    class AssignmentDueDate < DueDate
      scope :assignment_due_dates, ->(id) { where("deadline_type_id = ?", id) }

      ...
    end

    class TopicDueDate < DueDate
      scope :topic_due_dates, ->(id) { where("deadline_type_id = ?", id) }

      ...
    end
    ```

Remember that software design principles like SOLID, DRY etc. and patterns are not hard and fast rules, but they are guides to write more maintainable and flexible code. So, whether to adopt these suggestions or not depends on the functional and non-functional requirements of your application.
