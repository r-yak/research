```mermaid
classDiagram
    class User {
        #int: id
    }

    UserRepository o-- User : contains
    AbstractUser <|-- User : impl

```