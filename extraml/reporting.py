import os
import base64

class Reporter:
        def generate_html_report(self):
            # Create directories for assets if they don't exist
            for dir_name in ['report_assets/css', 'report_assets/js', 'report_assets/images']:
                os.makedirs(dir_name, exist_ok=True)

            # Extract Plotly JS code and save to a file
            plotly_js = next((item['content'] for item in self.report if item['content'].startswith('/*! For license information')), "")
            if plotly_js:
                with open('report_assets/js/plotly.min.js', 'w') as f:
                    f.write(plotly_js)

            # Extract CSS and save to a file
            css_content = """
            body {
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }
            h1 {
                color: #2c3e50;
                border-bottom: 2px solid #2c3e50;
                padding-bottom: 10px;
            }
            h2 {
                color: #34495e;
                margin-top: 30px;
            }
            pre {
                background-color: #f4f4f4;
                border: 1px solid #ddd;
                border-left: 3px solid #f36d33;
                color: #666;
                page-break-inside: avoid;
                font-family: monospace;
                font-size: 15px;
                line-height: 1.6;
                margin-bottom: 1.6em;
                max-width: 100%;
                overflow: auto;
                padding: 1em 1.5em;
                display: block;
                word-wrap: break-word;
            }
            img {
                max-width: 100%;
                height: auto;
                display: block;
                margin: 20px auto;
            }
            table {
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 1em;
            }
            th, td {
                text-align: left;
                padding: 8px;
                border-bottom: 1px solid #ddd;
            }
            tr:nth-child(even) {
                background-color: #f2f2f2;
            }
            .dataframe {
                border-collapse: collapse;
                margin: 25px 0;
                font-size: 0.9em;
                font-family: sans-serif;
                min-width: 400px;
                box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
            }
            .dataframe thead tr {
                background-color: #009879;
                color: #ffffff;
                text-align: left;
            }
            .dataframe th,
            .dataframe td {
                padding: 12px 15px;
            }
            .dataframe tbody tr {
                border-bottom: 1px solid #dddddd;
            }
            .dataframe tbody tr:nth-of-type(even) {
                background-color: #f3f3f3;
            }
            .dataframe tbody tr:last-of-type {
                border-bottom: 2px solid #009879;
            }
            div {
                overflow-x: auto;
            }
            """
            with open('report_assets/css/styles.css', 'w') as f:
                f.write(css_content)

            # Generate HTML content
            html_content = """
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>ExtraML Report</title>
                <link rel="stylesheet" href="report_assets/css/styles.css">
                <script src="report_assets/js/plotly.min.js"></script>
            </head>
            <body>
                <h1>ExtraML Report</h1>
            """

            for i, item in enumerate(self.report):
                html_content += f"<h2>{item['title']}</h2>"
                if item['content'].startswith('<img'):
                    # Extract base64 data and save as image file
                    img_data = item['content'].split(',')[1]
                    img_binary = base64.b64decode(img_data)
                    img_filename = f'report_assets/images/image_{i}.png'
                    with open(img_filename, 'wb') as f:
                        f.write(img_binary)
                    
                    # Use lazy loading for images
                    html_content += f'<img src="{img_filename}" alt="{item["title"]}" loading="lazy">'
                elif item['content'].startswith('<table'):
                    html_content += item['content']
                elif not item['content'].startswith('/*! For license information'):
                    html_content += f"<pre>{item['content']}</pre>"

            html_content += """
            </body>
            </html>
            """

            with open('extraml_report.html', 'w', encoding='utf-8') as f:
                f.write(html_content)

            print("HTML report generated: extraml_report.html")