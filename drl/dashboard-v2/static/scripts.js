/**
 * Created by bartkeulen on 7/6/17.
 */
$(document).ready(function() {

    $("#path").on("input", function(input) {
        Cookies.set("directory", input.currentTarget.value);
    });
});