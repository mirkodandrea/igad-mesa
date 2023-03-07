const GridLayoutModule = function (params) {
    const container = document.getElementById("elements");
    // set container style to use grid layout
    container.style.display = "grid";
    const templateRows = params["templateRows"];
    const templateCols = params["templateCols"];
    const gridAreas = params["gridAreas"];

    container.style.gridTemplateColumns = templateRows;
    container.style.gridTemplateRows = templateCols;

    this.render = (data) => {
        // loop childre of container and set grid area
        for (let i = 0; i < container.children.length; i++) {
            container.children[i].style.gridArea = gridAreas[i];

        }
    };
    this.reset = () => {};
};