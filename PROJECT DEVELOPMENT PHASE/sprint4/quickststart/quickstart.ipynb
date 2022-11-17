from __future__ import print_function

import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
import base64

from email import encoders
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
import mimetypes

import os
def verification():
# If modifying these scopes, delete the file token.json.
    SCOPES = ['https://mail.google.com/']


    def main():
        """Shows basic usage of the Gmail API.
        Lists the user's Gmail labels.
        """
        creds = None
        # The file token.json stores the user's access and refresh tokens, and is
        # created automatically when the authorization flow completes for the first
        # time.
        if os.path.exists('token.json'):
            creds = Credentials.from_authorized_user_file('token.json', SCOPES)
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', SCOPES)
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open('token.json', 'w') as token:
                token.write(creds.to_json())

        try:
            # Call the Gmail API
            service = build('gmail', 'v1', credentials=creds)
            results = service.users().labels().list(userId='me').execute()
            labels = results.get('labels', [])

            if not labels:
                print('No labels found.')
                return
            print('Labels:')
            for label in labels:
                print(label['name'])

        except HttpError as error:
            # TODO(developer) - Handle errors from gmail API.
            print(f'An error occurred: {error}')


    
    main()
if(not 'token.json' in os.listdir('.')):
    verification()
print('user verified. token is existing ')
def send_mail(to, subject, body, format='plain', attachments=[]):
    creds = None
    SCOPES = ['https://mail.google.com/']
    print(os.getcwd())
    creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    service = build('gmail', 'v1', credentials=creds)

    file_attachments = attachments

    #html = ''
    #with open('message.html') as msg:
    #    html += msg.read()

    #create email
    mimeMessage = MIMEMultipart()
    mimeMessage['to'] = to
    mimeMessage['subject'] = subject
    #mimeMessage.attach(MIMEText(html,'html'))
    mimeMessage.attach(MIMEText(body, format))

    for attachment in file_attachments:
        content_type, encoding = mimetypes.guess_type(attachment)
        main_type, sub_type = content_type.split('/', 1)
        file_name = os.path.basename(attachment)

        with open(attachment, 'rb') as f:
            myFile = MIMEBase(main_type, sub_type)
            myFile.set_payload(f.read())
            myFile.add_header('Content-Disposition', attachment, filename=file_name)
            encoders.encode_base64(myFile)

        mimeMessage.attach(myFile)


    raw_string = base64.urlsafe_b64encode(mimeMessage.as_bytes()).decode()


    message = service.users().messages().send(
        userId='me',
        body={'raw': raw_string}).execute()

    return message


#send_mail('','hi kk','it\'s nice meeting you :)')