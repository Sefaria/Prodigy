function raiseToTopByClassName(className) {
    console.log("raiseToTopByClassName");
    var elements = document.getElementsByClassName(className);

    if (elements.length > 0) {
        var element = elements[0];
        var container = element.parentNode;
        container.insertBefore(element, container.firstChild);
    } else {
        console.error('Element with class ' + className + ' not found.');
    }
}
function styleCheckedCheckboxes(styleObject) {
    var checkboxes = document.querySelectorAll('.c0197');
    console.log(checkboxes);

    checkboxes.forEach(function (checkbox) {
        // Apply each style property to the checked checkbox
        for (var property in styleObject) {
            if (styleObject.hasOwnProperty(property)) {
                checkbox.style[property] = styleObject[property];
            }
        }
    });
}

function  countCheckedCheckboxes(){
    let checkboxes = document.querySelectorAll('.c0197');
    console.log(checkboxes.length)
    return checkboxes.length;

}
// Function to check if an element with the given ID exists
function elementExists(id) {
    return document.getElementById(id) !== null;
}

// Function to remove an element by ID
function removeElementById(id) {
    var element = document.getElementById(id);
    if (element) {
        element.parentNode.removeChild(element);
    }
}
function addNumOfRecommendedElement(numOfRecommended){
    // Create a new section element
    var newSection = document.createElement('section');
    newSection.className = 'c0129 c0130';
    newSection.id = 'newSectionId';

    // Create an h3 element
    var h3Element = document.createElement('h3');
    h3Element.className = 'c0121 c0122';
    h3Element.textContent = 'Number of Recommended Labels: ' + numOfRecommended;

    // Create a span element
    var spanElement = document.createElement('span');

    // Append the h3 and span elements to the new section
    newSection.appendChild(h3Element);
    newSection.appendChild(spanElement);

    // Check if an element with the ID already exists
    if (elementExists('newSectionId')) {
    // If it exists, remove it
        removeElementById('newSectionId');
    }

    // Find the last section element on the page
    var sections = document.getElementsByTagName('section');
    var lastSection = sections[sections.length - 1];

    // Insert the new section after the last section
    lastSection.parentNode.insertBefore(newSection, lastSection.nextSibling);
}
document.addEventListener('prodigymount', function(event) {
  console.log("mounted");
  raiseToTopByClassName('prodigy-meta');
})
let changedColorForRecommended = false;
document.addEventListener('prodigyupdate', function(event) {
    if (!changedColorForRecommended) {
    styleCheckedCheckboxes({"accent-color": "red"});
    let numOfRecommended = countCheckedCheckboxes();
    addNumOfRecommendedElement(numOfRecommended);
    }
  changedColorForRecommended = true;
})
document.addEventListener('prodigyanswer', function(event) {
    styleCheckedCheckboxes({"accent-color": "red"});
    let numOfRecommended = countCheckedCheckboxes();
    addNumOfRecommendedElement(numOfRecommended);
})