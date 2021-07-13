import json
from datetime import datetime, timedelta
from os import path

import matplotlib.pyplot as plt
import numpy as np
import requests
from matplotlib.animation import FuncAnimation
import matplotlib.lines as mlines
from matplotlib.collections import PathCollection


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

    def __init__(self, covid_data):
        self.covid_data = covid_data
        self.fig, self.ax = plt.subplots()

        base = datetime.today()
        date_list = [base - timedelta(days=x) for x in range(GenAnimation.days, 0, -GenAnimation.step)]
        self.dates = [date.strftime("%Y-%m-%d") for date in date_list]

        self.stream = list(self.data_stream())

        self.ani = FuncAnimation(self.fig,
                                 self.update,
                                 frames=GenAnimation.days // GenAnimation.step,
                                 init_func=self.setup_plot,
                                 blit=True)

    def data_stream(self):
        for date in self.dates:

            # x-axis values
            xs = [county["data"][date]["vaccinationsCompletedRatio"] * 100
                  for county in self.covid_data
                  if county.get("data", {date: None}).get(date)]

            # y-axis values
            ys = [county["data"][date]["caseDensity"]
                  for county in self.covid_data
                  if county.get("data", {date: None}).get(date)]

            # size
            ss = [max(1, county["population"] / 10000) + 2
                  for county in self.covid_data
                  if county.get("data", {date: None}).get(date)]

            # weights
            ws = [county["population"]
                  for county in self.covid_data
                  if county.get("data", {date: None}).get(date)]

            cs = [county["color"]
                  for county in self.covid_data
                  if county.get("data", {date: None}).get(date)]
            yield xs, ys, ss, cs, ws

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        x, y, s, c, w = self.stream[0]
        self.ax.axis([5, 100, 0, 125])

        # x-axis label
        plt.xlabel('fully vaccinated %')
        # frequency label
        plt.ylabel('cases per 100K')
        # plot title
        plt.title('Covid US county case vs vaccine rate (with population and 2020 election)')
        # showing legend

        # best line fit
        line = give_me_a_straight_line(x, y, w)
        # best = self.ax.plot(x, line, 'r--', label='best fit trend-line')

        data = list(zip(x, y, s, c, w))
        # pick specific points to add to the legend
        dem = [point for point in data if point[3] == min(c)][0]
        rep = [point for point in data if point[3] == max(c)][0]
        _, mid_c = min([(abs(0.5 - color[0]), color) for color in c if color[1] == 0])
        neutral = [point for point in data if point[3] == mid_c][0]
        unknown = [point for point in data if point[3] == (0.5, 0.5, 0.5)][0]

        # add points to legend
        dems = self.ax.scatter([dem[0]], [dem[1]], [dem[2]], [dem[3]],
                               label="pop: {}0K, dem".format(int(dem[2]-2)))
        reps = self.ax.scatter([rep[0]], [rep[1]], [rep[2]], [rep[3]],
                               label="pop: {}0K, rep".format(int(rep[2]-2)))
        neutrals = self.ax.scatter([neutral[0]], [neutral[1]], [neutral[2]], [neutral[3]],
                                   label="pop: {}0K, neutral".format(int(neutral[2]-2)))
        unknowns = self.ax.scatter([unknown[0]], [unknown[1]], [unknown[2]], [unknown[3]],
                                   label="pop: {}0K, unknown".format(int(unknown[2]-2)))

        self.scat = self.ax.scatter(x, y, s=s, c=c)
        plt.legend()
        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,

    def update(self, frame):
        x, y, s, c, w = self.stream[frame]
        self.scat.set_offsets(np.c_[x, y])
        self.scat.set_sizes(np.array(s))
        self.scat.set_color(np.array(c))

        # best line fit
        ys = give_me_a_straight_line(x, y, w)
        line = mlines.Line2D(x, ys)  # , 'r--')
        transform = self.ax.transAxes
        line.set_transform(transform)
        # self.ax.add_line(line)
        return self.scat,


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


def main():

    election_data = get_processed_election_data()
    if election_data is None:
        election_data = json.loads(get_election_data())
        election_data = process_election_data(election_data)
    save_election_data(election_data)

    covid_data = get_covid_data()
    save_covid_data(covid_data)

    add_election_info_to_covid_data(covid_data, election_data)

    a = GenAnimation(covid_data)

    # a.setup_plot()
    # for _ in a.dates:
    #     a.next_plot()
    plt.show()


if __name__ == "__main__":
    main()
