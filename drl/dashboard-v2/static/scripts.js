/**
 * Created by bartkeulen on 7/6/17.
 */
$(document).ready(function() {

    // Initialize view
    set_view();

    // On hash change set correct view
    window.onhashchange = function() {
        set_view();
    };

    // Set active session cookie
    Cookies.set("active_sessions", JSON.stringify([]));

    // Set path cookie
    $("#path").on("input", function(input) {
        Cookies.set("directory", input.currentTarget.value);

        if (window.location.hash.substr(1) === "sessions") {
            set_view();
        }
    });

    $(document).on("click", ".row-clickable", function(event) {
        if (!$(event.target).hasClass("check-session")) {
            if ($(this).next().hasClass("runs-hidden")) {
                $(this).nextUntil(".row-clickable").removeClass("runs-hidden").addClass("runs-visible");
            } else {
                $(this).nextUntil(".row-clickable").removeClass("runs-visible").addClass("runs-hidden");

            }
        }
    });

    $(document).on("change", ".check-full-session", function() {
        $(this).parent().parent().nextUntil(".row-clickable").find(".check-run").prop("checked", this.checked);
        update_active_sessions();
    });

    $(document).on("change", ".check-run", function() {
        update_active_sessions();
    });
});

function set_view() {
    var anchor = document.location.hash.substr(1);
    if (anchor === "sessions") {
        get_summaries(function(data) {
            render_sessions(data);
        });
    }
    else if (anchor === "filters") {
        render_filters(null);
    }
    else {
        get_active_sessions(function(data) {
            var chart_ids = {};
            for (var i in data) {
                var env = data[i][2]["info"]["env"];
                if (!(chart_ids.hasOwnProperty(env))) {
                    chart_ids[env] = [];
                }

                var value_types = Object.keys(data[i][1]["values"]);
                for (var j in value_types) {
                    var id = (env + "_" + value_types[j]).split(" ").join("-");
                    if (chart_ids[env].indexOf(id) === -1) {
                        chart_ids[env].push(id);
                    }
                }
            }
            console.log(chart_ids);
            render_graphs(chart_ids);
            // update_chart_data(data);
        })
    }
}

function render_graphs(chart_ids) {
    $.get("static/graphs.hbs", function(source) {
        var template = Handlebars.compile(source);
        var html = template(chart_ids);
        $("#page-title").empty().append("Graphs");
        $('#main-content').empty().append(html);
    });
}

function render_sessions(data) {
    $.get("static/sessions.hbs", function(source) {
        var template = Handlebars.compile(source);
        var html = template(data);
        $("#page-title").empty().append("Sessions");
        $('#main-content').empty().append(html);
    });
}

function render_filters(data) {
    $.get("static/filters.hbs", function(source) {
        var template = Handlebars.compile(source);
        var html = template(data);
        $("#page-title").empty().append("Filters");
        $('#main-content').empty().append(html);
    });
}

// Get all summaries
function get_summaries(cb) {
    $.getJSON("/summaries", function (data) {
        cb(data);
    });
}

function update_active_sessions() {
    var paths = [];
    var count = $(".check-session").length;
    $(".check-session").each(function (idx) {
        if (this.checked) {
            if (!$(this).hasClass("check-full-session")) {
                paths.push($(this).parent().next().find(".path-session")[0].textContent);
            }
        }
        if (!--count) Cookies.set("active_sessions", JSON.stringify(paths));
    });
}

function get_active_sessions(cb) {
    var paths = JSON.parse(Cookies.get("active_sessions"));
    $.ajax({
        type: "POST",
        url: "/active_sessions",
        data:  JSON.stringify(paths),
        contentType: "application/json;charset=UTF-8",
        dataType: "json",
        success: function(response, status, jqXHR) {
            cb(response);
        },
        error: function(response, status, error) {
            alert("Error: " + error + ". Status: " + status);
            cb([]);
        }
    });
}

function update_chart_data(data) {
    TESTER = document.getElementById("chart");
    Plotly.plot(TESTER, [{
        x: [1, 2, 3, 4, 5],
        y: [1, 2, 4, 8, 16]
    }], {
	    margin: { t: 0 }
    });
}
