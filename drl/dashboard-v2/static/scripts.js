/**
 * Created by bartkeulen on 7/6/17.
 */
var UPDATE_INTERVAL = 5000;
var MA_PARAM = 5;
var interval = null;

$(document).ready(function() {
    // Set active session cookie
    if (!Cookies.get().hasOwnProperty("active_sessions")) {
        Cookies.set("active_sessions", JSON.stringify([]));
    }

    // Set moving average parameter cookie
    if (!Cookies.get().hasOwnProperty("ma_param")) {
        Cookies.set("ma_param", MA_PARAM);
    }

    // Set update interval parameter cookie
    if (!Cookies.get().hasOwnProperty("update_param")) {
        Cookies.set("update_param", UPDATE_INTERVAL);
    }

    // Initialize view
    set_view();

    // On hash change set correct view
    window.onhashchange = function() {
        set_view();
    };

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

    $(document).on("change", "#ma-param", function() {
        Cookies.set("ma_param", this.value);
    });

    $(document).on("change", "#update-param", function() {
        Cookies.set("update_param", this.value);
    });
});

function set_view() {
    if (interval != null) {
        clearInterval(interval);
    }
    var anchor = document.location.hash.substr(1);
    if (anchor === "sessions") {
        get_summaries(function(data) {
            render_sessions(data);
        });
    }
    else if (anchor === "filters") {
        var data = {
            "ma-param": Cookies.get("ma_param"),
            "update-param": Cookies.get("update_param")
        };
        render_filters(data);
    }
    else {
        get_active_sessions(function(data) {
            render_graphs(data, function() {
                update_chart_data(data);
                interval = setInterval(chart_update_loop, Cookies.get("update_param"));
            });
        });
    }
}

function render_graphs(chart_ids, cb) {
    $.get("static/graphs.hbs", function(source) {
        var template = Handlebars.compile(source);
        var html = template(chart_ids);
        $("#page-title").empty().append("Graphs");
        $('#main-content').empty().append(html);
        cb();
    });
}

function render_sessions(data) {
    console.log(data);
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
                paths.push(this.id);
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
            cb(convert_data_to_chart(response));
        },
        error: function(response, status, error) {
            alert("Error: " + error + ". Status: " + status);
            cb([]);
        }
    });
}

function convert_data_to_chart(data) {
    var charts = {};
    for (var i in data) {
        var env = data[i][2]["info"]["env"];
        if (!charts.hasOwnProperty(env)) {
            charts[env] = [];
        }

        var value_types = Object.keys(data[i][1]["values"]);
        for (var j in value_types) {
            var id = get_chart_id(env, value_types[j]);
            var x = data[i][1]["episode"];
            var y = data[i][1]["values"][value_types[j]];
            var filt = moving_average(x, y);
            var name = data[i][2]["info"]["name"] + " - " + data[i][2]["info"]["timestamp"] + " - " + data[i][2]["info"]["run"];

            var values = {
                x: filt[0],
                y: filt[1],
                name: name,
                mode: 'lines',
                line: {width: 1}
            };

            var exists = false;
            for (var k in charts[env]) {
                if (charts[env][k]["id"] === id) {
                    charts[env][k]["values"].push(values);
                    exists = true;
                }
            }
            if (!exists) {
                charts[env].push({"id": id, "values": [values]});
            }
        }
    }
    return charts;
}

function update_chart_data(charts) {
    for (var env in charts) {
        for (var i in charts[env]) {
            var elem = document.getElementById(charts[env][i]["id"]);

            var values = charts[env][i]["values"];

            var layout = {
                title: charts[env][i]["id"],
                showlegend: true,
                legend: {"orientation": "h"}
            };

            Plotly.newPlot(elem, values, layout);
        }
    }
}

function chart_update_loop() {
    get_active_sessions(function(charts) {
        for (var env in charts) {
            for (var i in charts[env]) {
                var elem = document.getElementById(charts[env][i]["id"]);
                elem.data = charts[env][i]["values"];
                Plotly.redraw(elem);
            }
        }
    });
}

function get_chart_id(env, value_type) {
    return (env + "_" + value_type).split(" ").join("-");
}

function moving_average(episode, data) {
    var new_episode = [];
    var new_data = [];
    var n = Cookies.get("ma_param");
    for (var i = 0; i < data.length-n+1; i++) {
        var sum = 0.;
        for (var j = 0; j < n; j++) {
            sum += data[i+j];
        }
        new_data.push(sum/n);
        new_episode.push(episode[i])
    }
    return [new_episode, new_data];
}