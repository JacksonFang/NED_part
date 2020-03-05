import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import base64

passwd = 'ZGdhYjMxNzU='
p = base64.b64decode(passwd).decode('utf-8')


mail_content = '''Emerging security news in 2019-08-23 ~ 2019-09-17'''

# The mail addresses and password
sender_address = 'a3322683@gmail.com'
sender_pass = p
receiver_address = 'a3322683@ids.mis.nsysu.edu.tw'
# Setup the MIME
message = MIMEMultipart()
message['From'] = sender_address
message['To'] = receiver_address
message['Subject'] = 'security news of 2019-08-23 ~ 2019-09-17 '
# The subject line
# The body and the attachments for the mail
message.attach(MIMEText(mail_content, 'plain'))

# # attach_file_name = 'result.xlsx'
# attach_file_name = 'document_cluster'
# attach_file = open(attach_file_name, 'rb')  # Open the file as binary mode
# payload = MIMEBase('application', 'octet-stream')
# payload.set_payload((attach_file).read())
# encoders.encode_base64(payload)  # encode the attachment
# # add payload header with filename
# payload.add_header('Content-Decomposition', 'attachment', filename=attach_file_name)
# message.attach(payload)

a = 'hello' +'.xlsx'
part = MIMEBase('application', "octet-stream")
part.set_payload(open("result.xlsx", "rb").read())
# part.set_payload(open("document_cluster", "rb").read())
encoders.encode_base64(part)
part.add_header('Content-Disposition', 'attachment; filename="news list.xlsx"')
message.attach(part)


# Create SMTP session for sending the mail
session = smtplib.SMTP('smtp.gmail.com', 587)  # use gmail with port
session.starttls()  # enable security
session.login(sender_address, sender_pass)  # login with mail_id and password
text = message.as_string()
session.sendmail(sender_address, receiver_address, text)
session.quit()
print('Mail Sent')
