#!/usr/bin/env python
# pylint: disable=unused-argument

import logging
import os
import time
from datetime import datetime
from telegram import ForceReply, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from dotenv import load_dotenv
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
from img_processing import download_and_preprocess_image, ensure_output_dir
load_dotenv()

# Load environment variables
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
AZURE_VISION_KEY = os.getenv("COMPUTER_VISION_KEY")
AZURE_VISION_ENDPOINT = os.getenv("COMPUTER_VISION_ENDPOINT")

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Initialize Azure Computer Vision client
computervision_client = ComputerVisionClient(
    AZURE_VISION_ENDPOINT,
    CognitiveServicesCredentials(AZURE_VISION_KEY)
)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    await update.message.reply_html(
        rf"Hi {user.mention_html()}! Send me an image with handwritten text, and I'll convert it to digital text!",
        reply_markup=ForceReply(selective=True),
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text(
        "Just send me a photo containing handwritten text, and I'll extract the text for you!"
    )

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle received photos and extract text using Azure Computer Vision."""
    try:
        # Send initial processing message
        processing_message = await update.message.reply_text(
            "🔍 Receiving your image..."
        )

        # Get the photo file
        photo = update.message.photo[-1]  # Get the largest photo size
        file = await context.bot.get_file(photo.file_id)
        photo_url = file.file_path
        
        await processing_message.edit_text("🖼️ Preprocessing image to enhance text recognition...")
        
        # Create output directory
        output_dir = ensure_output_dir()
        
        # Preprocess the image
        try:
            processed_image = download_and_preprocess_image(photo_url)
            await processing_message.edit_text(
                f"📥 Image enhanced! Images saved in '{output_dir}' folder. Sending to Azure..."
            )
        except Exception as e:
            logger.error(f"Preprocessing error: {str(e)}")
            await processing_message.edit_text("⚠️ Error in preprocessing, trying with original image...")
            processed_image = None

        # Call Azure's OCR service
        if processed_image:
            read_response = computervision_client.read_in_stream(processed_image, raw=True)
        else:
            read_response = computervision_client.read(photo_url, raw=True)
        # Get the operation location (URL with ID of the operation)
        read_operation_location = read_response.headers["Operation-Location"]
        operation_id = read_operation_location.split("/")[-1]

        await processing_message.edit_text("🔎 Azure is analyzing the text...")

        # Wait for the operation to complete
        dots = 1
        while True:
            read_result = computervision_client.get_read_result(operation_id)
            if read_result.status not in ['notStarted', 'running']:
                break
            
            # Alternate between different waiting messages
            dots = (dots % 3) + 1
            waiting_message = f"🔎 Analyzing your image{'.' * dots}"
            
            try:
                await processing_message.edit_text(waiting_message)
            except Exception as e:
                logger.debug(f"Message edit failed (expected): {str(e)}")
                
            time.sleep(1)

        # Extract the text
        if read_result.status == OperationStatusCodes.succeeded:
            text_results = []
            for text_result in read_result.analyze_result.read_results:
                for line in text_result.lines:
                    text_results.append(line.text)
            
            if text_results:
                # Join all lines with newlines
                extracted_text = "\n".join(text_results)
                
                # Send the final result
                await processing_message.edit_text(
                    f"✨ Here's the extracted text:\n\n{extracted_text}"
                )
                logger.info("Successfully extracted text from image")
            else:
                await processing_message.edit_text(
                    "❌ No text was found in the image. Please try with another image that contains clear handwritten text!"
                )
        else:
            await processing_message.edit_text(
                "❌ Sorry, I couldn't process the text in this image. Please try with another image!"
            )

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        try:
            await processing_message.edit_text(
                "❌ Sorry, an error occurred while processing your image. Please try again!"
            )
        except Exception:
            await update.message.reply_text(
                "❌ Sorry, an error occurred while processing your image. Please try again!"
            )

def main() -> None:
    """Start the bot."""
    # Verify required environment variables
    if not all([TELEGRAM_TOKEN, AZURE_VISION_KEY, AZURE_VISION_ENDPOINT]):
        logger.error("Missing required environment variables!")
        return

    # Create the Application and pass it your bot's token
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
