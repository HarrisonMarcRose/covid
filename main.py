import json
from datetime import datetime, timedelta
from os import path

import matplotlib.pyplot as plt
import numpy as np
import requests
from matplotlib.animation import FuncAnimation


def give_me_a_straight_line(xs, ys, ws):
    w, b = np.polyfit(xs, ys, w=ws, deg=1)
    line = [w * x + b for x in xs]
    return line


def save_covid_data(covid_data):
    if not path.exists('covidactnow-{}.timeseries.json'.format(datetime.today().strftime("%Y-%m-%d"))):
        with open('covidactnow-{}.timeseries.json'.format(datetime.today().strftime("%Y-%m-%d")), 'w') as file:
            return file.write(json.dumps(covid_data))

    return None


def save_election_data(election_data):
    if not path.exists('2020-election-data.json'):
        with open('2020-election-data.json', 'w') as file:
            return file.write(json.dumps(election_data))

    return None


def get_processed_election_data():
    if path.exists('2020-election-data.json'):
        with open('2020-election-data.json') as file:
            return json.loads(file.read())

    return None


def get_election_data():
    with open('precincts-with-results.geojson') as file:
        return file.read()


def get_covid_data():
    if path.exists('covidactnow-{}.timeseries.json'.format(datetime.today().strftime("%Y-%m-%d"))):
        with open('covidactnow-{}.timeseries.json'.format(datetime.today().strftime("%Y-%m-%d"))) as file:
            return json.loads(file.read())

    response = requests.get(
        url="https://api.covidactnow.org/v2/counties.timeseries.json",
        params={"apiKey": "ac42d1aa586945cb97b1e8cbaa32a02f"})

    data = []
    for county in response.json():
        data.append(
            {
                "state": county["state"],
                "county": county["county"],
                "fips": county["fips"],
                "population": county["population"],
                "data": dict()
            }

        )

        for item in county["metricsTimeseries"]:
            if item.get("vaccinationsCompletedRatio") is not None and \
                    item.get("caseDensity") is not None:
                data[-1]["data"].update({
                    item["date"]:
                        {
                            "caseDensity": item["caseDensity"],
                            "vaccinationsCompletedRatio": item["vaccinationsCompletedRatio"],
                            "vaccinationsInitiatedRatio": item["vaccinationsInitiatedRatio"]
                        }
                })

    return data


class GenAnimation:
    days = 180
    step = 1

    def __init__(self, stream):
        self.fig, self.ax = plt.subplots()
        self.stream = stream
        self.ani = FuncAnimation(self.fig,
                                 self.update,
                                 frames=GenAnimation.days // GenAnimation.step,
                                 init_func=self.setup_plot,
                                 blit=True)

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        x, y, s, c, w = self.stream[0]
        self.ax.axis([0, 100, 0, 125])

        # x-axis label
        plt.xlabel('fully vaccinated %')
        # frequency label
        plt.ylabel('cases per 100K')
        # plot title
        plt.title('Covid US county case vs vaccine rate (with population and 2020 election)')
        # showing legend

        # best line fit
        ys = give_me_a_straight_line(x, y, w)
        self.line, = self.ax.plot(x, ys, 'r--', label='best fit trend-line')

        data = list(zip(x, y, s, c, w))
        # pick specific points to add to the legend
        dem = [point for point in data if point[3] == min(c)][0]
        print(data.index(dem))
        rep = [point for point in data if point[3] == max(c)][0]
        print(data.index(rep))
        _, mid_c = min([(abs(0.5 - color[0]), color) for color in c if color[1] == 0])
        neutral = [point for point in data if point[3] == mid_c][0]
        print(data.index(neutral))
        unknown = [point for point in data if point[3] == (0.5, 0.5, 0.5)][0]
        print(data.index(unknown))

        # add points to legend
        self.dems = self.ax.scatter([dem[0]], [dem[1]], [dem[2]], [dem[3]],
                               label="pop: {}0K, dem".format(int(dem[2]-2)))
        self.reps = self.ax.scatter([rep[0]], [rep[1]], [rep[2]], [rep[3]],
                               label="pop: {}0K, rep".format(int(rep[2]-2)))
        self.neutrals = self.ax.scatter([neutral[0]], [neutral[1]], [neutral[2]], [neutral[3]],
                                   label="pop: {}0K, neutral".format(int(neutral[2]-2)))
        self.unknowns = self.ax.scatter([unknown[0]], [unknown[1]], [unknown[2]], [unknown[3]],
                                   label="pop: {}0K, unknown".format(int(unknown[2]-2)))

        self.scat = self.ax.scatter(x, y, s=s, c=c)
        self.legend = plt.legend()
        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat, self.line, self.dems, self.reps, self.neutrals, self.unknowns

    def update(self, frame):
        x, y, s, c, w = self.stream[frame]

        # data = list(zip(x, y, s, c, w))
        # dem = [point for point in data if point[3] == min(c)][0]
        # dem_index = data.index(dem)
        # rep = [point for point in data if point[3] == max(c)][0]
        # rep_index = data.index(rep)
        # _, mid_c = min([(abs(0.5 - color[0]), color) for color in c if color[1] == 0])
        # neutral = [point for point in data if point[3] == mid_c][0]
        # neutral_index = data.index(neutral)
        # unknown = [point for point in data if point[3] == (0.5, 0.5, 0.5)][0]
        # unknown_index = data.index(unknown)
        # self.scat.set_offsets(np.c_[x[dem_index], y[dem_index]])
        # self.scat.set_sizes(np.array(s[dem_index]))
        # self.scat.set_color(np.array(c[dem_index]))
        # self.fig.legend(handles=(self.scat,),
        #                 labels=("pop: {}0K, dem".format(int(dem[2] - 2)),))

        self.scat.set_offsets(np.c_[x, y])
        self.scat.set_sizes(np.array(s))
        self.scat.set_color(np.array(c))

        # best line fit
        ys = give_me_a_straight_line(x, y, w)
        self.line.set_xdata(x)
        self.line.set_ydata(ys)
        self.line.set_label('best fit trend-line')

        return self.scat, self.line


def add_election_info_to_covid_data(covid_data, election_data):
    # add margin info for color
    for d in covid_data:
        if election_data.get(d["fips"]):
            d["pct_dem_lead"] = (
                    (
                            election_data[d["fips"]]["votes_dem"] -
                            election_data[d["fips"]]["votes_rep"]
                    ) /
                    election_data[d["fips"]]["votes_total"]
            )
            d["color"] = (
                min(max(0.5 - d["pct_dem_lead"], 0), 1),
                0,
                max(min(0.5 + d["pct_dem_lead"], 1), 0)
            )
        else:
            d["pct_dem_lead"] = None
            d["color"] = (0.5, 0.5, 0.5)


def process_election_data(election_data):
    """Combine precints into a single county"""

    precints = [precint['properties'] for precint in election_data['features']]
    county_data = dict()
    for precint in precints:

        if precint["votes_dem"] is None \
                or precint["votes_rep"] is None \
                or precint["votes_total"] is None:
            continue

        fips = precint['GEOID'][:5]
        processed_county = county_data.get(fips)
        if processed_county is None:
            county_data[fips] = {
                "votes_dem": precint["votes_dem"],
                "votes_rep": precint["votes_rep"],
                "votes_total": precint["votes_total"],
                "precincts": 1

            }
        else:
            county_data[fips] = {
                "votes_dem": precint["votes_dem"] + county_data[fips]["votes_dem"],
                "votes_rep": precint["votes_rep"] + county_data[fips]["votes_rep"],
                "votes_total": precint["votes_total"] + county_data[fips]["votes_total"],
                "precincts": 1 + county_data[fips]["precincts"]
            }
    return county_data


def gen_plot_data(covid_data):
    base = datetime.today()
    date_list = [base - timedelta(days=x) for x in range(GenAnimation.days, 0, -GenAnimation.step)]
    dates = [date.strftime("%Y-%m-%d") for date in date_list]

    def data_stream():
        for date in dates:

            # x-axis values
            xs = [county["data"][date]["vaccinationsCompletedRatio"] * 100
                  for county in covid_data
                  if county.get("data", {date: None}).get(date)]

            # y-axis values
            ys = [county["data"][date]["caseDensity"]
                  for county in covid_data
                  if county.get("data", {date: None}).get(date)]

            # size
            ss = [max(1, county["population"] / 10000) + 2
                  for county in covid_data
                  if county.get("data", {date: None}).get(date)]

            # weights
            ws = [county["population"]
                  for county in covid_data
                  if county.get("data", {date: None}).get(date)]

            cs = [county["color"]
                  for county in covid_data
                  if county.get("data", {date: None}).get(date)]
            yield xs, ys, ss, cs, ws

    return list(data_stream())


def main():

    election_data = get_processed_election_data()
    if election_data is None:
        election_data = json.loads(get_election_data())
        election_data = process_election_data(election_data)
    save_election_data(election_data)

    covid_data = get_covid_data()
    save_covid_data(covid_data)

    add_election_info_to_covid_data(covid_data, election_data)
    stream_data = gen_plot_data(covid_data)

    # give a array of subplots for data going back in time
    # fig = plt.figure()
    # rows = 3
    # cols = 3
    # step_days = 14
    # gs = fig.add_gridspec(rows, cols, hspace=0, wspace=0, left=0.05, right=0.95, bottom=0.05, top=0.95)
    # axs = gs.subplots(sharex=True, sharey=True)
    # for i in range(rows):
    #     for j in range(cols):
    #         index = - 1 - step_days * i * cols - step_days * j
    #         print(i, j, index)
    #         axs[i, j].label_outer()
    #         axs[i, j].axis([0, 100, 0, 150])
    #         axs[i, j].scatter(
    #             stream_data[index][0],
    #             stream_data[index][1],
    #             s=stream_data[index][2],
    #             c=stream_data[index][3])
    #
    #         # best line fit
    #         ys = give_me_a_straight_line(stream_data[index][0], stream_data[index][1], stream_data[index][4])
    #         axs[i, j].plot(stream_data[index][0], ys, 'r--', label='best fit trend-line')
    #
    # plt.show()

    animation = GenAnimation(stream_data)
    plt.show()


if __name__ == "__main__":
    main()
