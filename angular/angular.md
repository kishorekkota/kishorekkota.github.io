
## What is a Component in Angular ?

/**
 * This component represents a basic Angular component.
 * It serves as a building block for creating user interface elements in an Angular application.
 * Components are responsible for managing a part of the application's view and logic.
 * They encapsulate the HTML template, CSS styles, and behavior associated with a specific part of the user interface.
 * Components can be reused and composed together to create complex UI structures.
 */
@Component({
    selector: 'app-example',
    templateUrl: './example.component.html',
    styleUrls: ['./example.component.css']
})
export class ExampleComponent implements OnInit {
    // Component properties and methods go here
    // ...
    
    ngOnInit() {
        // Initialization logic goes here
        // ...
    }
}