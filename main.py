import json
from datetime import datetime
from os import path

import matplotlib.pyplot as plt
import numpy as np
import requests


def give_me_a_straight_line(xs, ys, ws):
    w, b = np.polyfit(xs, ys, w=ws, deg=1)
    line = [w * x + b for x in xs]
    return line


def save_covid_data(covid_data):
    if not path.exists('covidactnow-{}.json'.format(datetime.today().strftime("%Y-%m-%d"))):
        with open('covidactnow-{}.json'.format(datetime.today().strftime("%Y-%m-%d")), 'w') as file:
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
    if path.exists('covidactnow-{}.json'.format(datetime.today().strftime("%Y-%m-%d"))):
        with open('covidactnow-{}.json'.format(datetime.today().strftime("%Y-%m-%d"))) as file:
            return json.loads(file.read())

    response = requests.get(
        url="http://api.covidactnow.org/v2/counties.json",
        params={"apiKey": "ac42d1aa586945cb97b1e8cbaa32a02f"})
    data = [
        {
            "state": county["state"],
            "county": county["county"],
            "fips": county["fips"],
            "population": county["population"],
            "caseDensity": county["metrics"]["caseDensity"],
            "vaccinationsCompletedRatio": county["metrics"]["vaccinationsCompletedRatio"],
            "vaccinationsInitiatedRatio": county["metrics"]["vaccinationsInitiatedRatio"]
        }
        for county in response.json()
        if county["metrics"]["vaccinationsCompletedRatio"] is not None
        and county["metrics"]["caseDensity"] is not None
    ]

    return data


def generate_plot(covid_data):
    # x-axis values
    xs = [county["vaccinationsCompletedRatio"] * 100 for county in covid_data]
    # y-axis values
    ys = [county["caseDensity"] for county in covid_data]
    # size
    ss = [max(1, county["population"] / 10000) for county in covid_data]
    # weights
    ws = [county["population"] for county in covid_data]
    # colors
    cs = [county["color"] for county in covid_data]
    # labels
    # ls = ["pop: {} 10K, dem: {}".format(
    #         int(max(1, county["population"]/10000)),
    #         "NA" if county["pct_dem_lead"] is None else "{:.1f}".format(county["pct_dem_lead"]))
    #       for county in data]
    # plotting points as a scatter plot
    # for x, y, s, c, l in zip(xs, ys, ss, cs, ls):
    plt.scatter(xs, ys, s=ss, c=cs)
    max_dem = max([county["pct_dem_lead"] for county in covid_data
                   if county["pct_dem_lead"] and county["population"] > 50000])
    dem = [county for county in covid_data if county["pct_dem_lead"] == max_dem][0]
    max_rep = min([county["pct_dem_lead"] for county in covid_data
                   if county["pct_dem_lead"] and county["population"] > 50000])
    rep = [county for county in covid_data if county["pct_dem_lead"] == max_rep][0]
    neutral = [county for county in covid_data
               if county["pct_dem_lead"] and county["population"] > 50000
               and abs(county["pct_dem_lead"]) < 0.01][0]
    unknown = [county for county in covid_data
               if county["pct_dem_lead"] is None and county["population"] > 50000][0]
    # pick specific points to add to the legend
    plt.scatter([dem["vaccinationsCompletedRatio"] * 100],
                [dem["caseDensity"]],
                s=[max(1, dem["population"] / 10000)],
                c=[dem["color"]],
                label="pop: {}0K, dem: {}".format(
                    int(max(1, dem["population"] / 10000)),
                    "{:.1%}".format(dem["pct_dem_lead"])
                ))
    plt.scatter([rep["vaccinationsCompletedRatio"] * 100],
                [rep["caseDensity"]],
                s=[max(1, rep["population"] / 10000)],
                c=[rep["color"]],
                label="pop: {}0K, dem: {}".format(
                    int(max(1, rep["population"] / 10000)),
                    "{:.1%}".format(rep["pct_dem_lead"])
                ))
    plt.scatter([neutral["vaccinationsCompletedRatio"] * 100],
                [neutral["caseDensity"]],
                s=[max(1, neutral["population"] / 10000)],
                c=[neutral["color"]],
                label="pop: {}0K, dem: {}".format(
                    int(max(1, neutral["population"] / 10000)),
                    "{:.1%}".format(neutral["pct_dem_lead"])
                ))
    plt.scatter([unknown["vaccinationsCompletedRatio"] * 100],
                [unknown["caseDensity"]],
                s=[max(1, unknown["population"] / 10000)],
                c=[unknown["color"]],
                label="pop: {}0K, dem: {}".format(
                    int(max(1, unknown["population"] / 10000)),
                    "unknown"
                ))
    line = give_me_a_straight_line(xs, ys, ws)
    plt.plot(xs, line, 'r--', label='best fit trend-line')

    # x-axis label
    plt.xlabel('fully vaccinated %')
    # frequency label
    plt.ylabel('cases per 100K')
    # plot title
    plt.title('Covid US county case vs vaccine rate (with population and 2020 election)')
    # showing legend
    plt.legend()  # labels=labels)
    # function to show the plot
    plt.show()


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

    generate_plot(covid_data)


if __name__ == "__main__":
    main()
